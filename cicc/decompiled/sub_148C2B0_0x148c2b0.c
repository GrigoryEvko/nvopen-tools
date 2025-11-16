// Function: sub_148C2B0
// Address: 0x148c2b0
//
__int64 __fastcall sub_148C2B0(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rcx
  int v7; // edi
  __int64 v9; // rsi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 v19; // rsi
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v23; // r14
  char v24; // al
  _QWORD *v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r10
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned int v31; // edi
  __int64 v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 *v35; // r12
  __int64 v36; // rdi
  __int64 v37; // rax
  bool v38; // al
  __int64 *v39; // rbx
  int v40; // eax
  int v41; // r9d
  __int64 v42; // rbx
  __int64 v43; // r12
  int v44; // eax
  unsigned int v45; // esi
  int v46; // eax
  __int64 v47; // rax
  int v48; // r9d
  __int64 v49; // r12
  unsigned __int8 v50; // al
  _QWORD *v51; // rax
  __int64 v52; // r15
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp+30h] [rbp-130h]
  __int64 v57; // [rsp+38h] [rbp-128h]
  __int64 v58; // [rsp+40h] [rbp-120h]
  __int64 v59; // [rsp+50h] [rbp-110h]
  __int64 v61; // [rsp+60h] [rbp-100h]
  __int64 v62; // [rsp+68h] [rbp-F8h]
  unsigned int v63; // [rsp+68h] [rbp-F8h]
  unsigned int v64; // [rsp+68h] [rbp-F8h]
  void *v65; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v66; // [rsp+78h] [rbp-E8h] BYREF
  __int64 v67; // [rsp+88h] [rbp-D8h]
  __int64 v68; // [rsp+90h] [rbp-D0h]
  _QWORD *v69; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-B8h] BYREF
  __int64 v71; // [rsp+B0h] [rbp-B0h]
  __int64 v72; // [rsp+B8h] [rbp-A8h]
  int v73; // [rsp+C0h] [rbp-A0h]
  __int64 v74; // [rsp+C8h] [rbp-98h]
  __int64 v75; // [rsp+D0h] [rbp-90h]
  bool v76; // [rsp+D8h] [rbp-88h]
  __int64 *v77; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v78; // [rsp+E8h] [rbp-78h] BYREF
  __int64 v79; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v80; // [rsp+F8h] [rbp-68h]
  __int64 v81; // [rsp+100h] [rbp-60h]
  __int64 v82; // [rsp+108h] [rbp-58h]
  __int16 v83; // [rsp+110h] [rbp-50h]

  v4 = *(_QWORD *)(a1 + 64);
  v5 = *(_DWORD *)(v4 + 24);
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(a2 + 40);
  v7 = v5 - 1;
  v9 = *(_QWORD *)(v4 + 8);
  v10 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v11 = (__int64 *)(v9 + 16LL * v10);
  v12 = *v11;
  if ( v6 != *v11 )
  {
    v40 = 1;
    while ( v12 != -8 )
    {
      v41 = v40 + 1;
      v10 = v7 & (v40 + v10);
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v6 == *v11 )
        goto LABEL_3;
      v40 = v41;
    }
    return 0;
  }
LABEL_3:
  v13 = v11[1];
  v59 = v13;
  if ( !v13 || **(_QWORD **)(v13 + 32) != v6 || (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 0 )
    return 0;
  v14 = v13 + 56;
  v15 = 0;
  v61 = 0;
  v62 = 8LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v16 = a2;
  v17 = 0;
  v18 = v16;
  do
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v18 + 23) & 0x40) != 0 )
        v19 = *(_QWORD *)(v18 - 8);
      else
        v19 = v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF);
      v20 = *(_QWORD *)(v19 + 3 * v17);
      if ( sub_1377F70(v14, *(_QWORD *)(v17 + v19 + 24LL * *(unsigned int *)(v18 + 56) + 8)) )
      {
        if ( v15 )
        {
          if ( v20 != v15 )
            return 0;
        }
        else
        {
          v15 = v20;
        }
        goto LABEL_11;
      }
      if ( !v61 )
        break;
      if ( v20 != v61 )
        return 0;
LABEL_11:
      v17 += 8;
      if ( v62 == v17 )
        goto LABEL_19;
    }
    v61 = v20;
    v17 += 8;
  }
  while ( v62 != v17 );
LABEL_19:
  v23 = v18;
  if ( v61 == 0 || v15 == 0 )
    return 0;
  v21 = sub_147EA10(a1, v18, v15, v61);
  if ( v21 )
    return v21;
  v57 = sub_145DC80(a1, v18);
  v56 = a1 + 144;
  sub_1457D90(&v65, v18, a1);
  v79 = 0;
  v78 = v66 & 6;
  v80 = v67;
  if ( v67 != 0 && v67 != -8 && v67 != -16 )
    sub_1649AC0(&v78, v66 & 0xFFFFFFFFFFFFFFF8LL);
  v77 = (__int64 *)&unk_49EC5C8;
  v81 = v68;
  v82 = v57;
  v24 = sub_145F6E0(v56, (__int64)&v77, &v69);
  v25 = v69;
  if ( !v24 )
  {
    v44 = *(_DWORD *)(a1 + 160);
    v45 = *(_DWORD *)(a1 + 168);
    ++*(_QWORD *)(a1 + 144);
    v46 = v44 + 1;
    if ( 4 * v46 >= 3 * v45 )
    {
      v45 *= 2;
    }
    else if ( v45 - *(_DWORD *)(a1 + 164) - v46 > v45 >> 3 )
    {
LABEL_64:
      *(_DWORD *)(a1 + 160) = v46;
      sub_1457D90(&v69, -8, 0);
      v47 = v72;
      if ( v72 != v25[3] )
        --*(_DWORD *)(a1 + 164);
      v69 = &unk_49EE2B0;
      if ( v47 != -8 && v47 != 0 && v47 != -16 )
        sub_1649B30(&v70);
      sub_1453650((__int64)(v25 + 1), &v78);
      v25[4] = v81;
      v25[5] = v82;
      goto LABEL_25;
    }
    sub_14676C0(v56, v45);
    sub_145F6E0(v56, (__int64)&v77, &v69);
    v25 = v69;
    v46 = *(_DWORD *)(a1 + 160) + 1;
    goto LABEL_64;
  }
LABEL_25:
  v77 = (__int64 *)&unk_49EE2B0;
  sub_1455FA0((__int64)&v78);
  v65 = &unk_49EE2B0;
  sub_1455FA0((__int64)&v66);
  v26 = sub_146F1B0(a1, v15);
  v27 = v26;
  if ( *(_WORD *)(v26 + 24) != 4 )
  {
    v78 = 0;
    v79 = 0;
    v77 = (__int64 *)a1;
    v80 = 0;
    LODWORD(v81) = 0;
    v82 = v59;
    LOBYTE(v83) = 1;
    v42 = sub_148A520((__int64 *)&v77, v26, a3, a4);
    if ( !(_BYTE)v83 )
      v42 = sub_1456E90(a1);
    j___libc_free_0(v79);
    v78 = 0;
    v77 = (__int64 *)a1;
    v79 = 0;
    v82 = v59;
    v80 = 0;
    LODWORD(v81) = 0;
    v83 = 0;
    v43 = sub_148BD30(&v77, v42, a3, a4);
    if ( v83 )
      v43 = sub_1456E90(a1);
    j___libc_free_0(v79);
    if ( v42 != sub_1456E90(a1) && v43 != sub_1456E90(a1) && v43 == sub_146F1B0(a1, v61) )
    {
      sub_1464870(a1, v23, v57);
      sub_1457D90(&v77, v23, a1);
      v21 = v42;
      sub_1467C40(v56, &v77)[5] = v42;
      v77 = (__int64 *)&unk_49EE2B0;
      sub_1455FA0((__int64)&v78);
      return v21;
    }
    goto LABEL_60;
  }
  v28 = *(_QWORD *)(v26 + 40);
  v63 = v28;
  if ( (_DWORD)v28 )
  {
    v29 = *(_QWORD *)(v26 + 32);
    v30 = 0;
    do
    {
      if ( v57 == *(_QWORD *)(v29 + 8 * v30) )
      {
        v63 = v30;
        goto LABEL_31;
      }
      v31 = ++v30;
    }
    while ( (unsigned int)v28 != v30 );
    v63 = v31;
    v30 = v31;
  }
  else
  {
    v30 = 0;
  }
LABEL_31:
  if ( v28 == v30 )
  {
LABEL_60:
    sub_1464220(a1, v23);
    return v21;
  }
  v32 = 0;
  v77 = &v79;
  v78 = 0x800000000LL;
  v33 = *(_QWORD *)(v27 + 40);
  v58 = (unsigned int)v33;
  if ( (_DWORD)v33 )
  {
    v34 = v27;
    while ( v63 == (_DWORD)v32 )
    {
LABEL_41:
      if ( v58 == ++v32 )
      {
        v21 = 0;
        goto LABEL_43;
      }
    }
    v35 = *(__int64 **)(*(_QWORD *)(v34 + 32) + 8 * v32);
    v36 = sub_13FCB50(v59);
    if ( v36 )
    {
      v37 = sub_157EBA0(v36);
      if ( *(_BYTE *)(v37 + 16) != 26 || (*(_DWORD *)(v37 + 20) & 0xFFFFFFF) != 3 )
        goto LABEL_40;
      v36 = *(_QWORD *)(v37 - 72);
      v38 = **(_QWORD **)(v59 + 32) == *(_QWORD *)(v37 - 24);
    }
    else
    {
      v38 = 0;
    }
    v75 = v36;
    v70 = 0;
    v69 = (_QWORD *)a1;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    v74 = v59;
    v76 = v38;
    v35 = sub_1489EA0((__int64 *)&v69, (__int64)v35, a3, a4);
    j___libc_free_0(v71);
LABEL_40:
    v69 = v35;
    sub_1458920((__int64)&v77, &v69);
    goto LABEL_41;
  }
LABEL_43:
  v39 = sub_147DD40(a1, (__int64 *)&v77, 0, 0, a3, a4);
  if ( !sub_146CEE0(a1, (__int64)v39, v59) && (*((_WORD *)v39 + 12) != 7 || v39[6] != v59) )
  {
    if ( v77 != &v79 )
      _libc_free((unsigned __int64)v77);
    goto LABEL_60;
  }
  sub_1455040((__int64)&v69, v15, *(_QWORD *)(a1 + 56));
  if ( (_BYTE)v74 )
  {
    v48 = 0;
    if ( (_DWORD)v69 == 11 && v70 == v23 )
    {
      v48 = 2 * (BYTE1(v72) != 0);
      if ( (_BYTE)v72 )
        v48 |= 4u;
    }
    goto LABEL_73;
  }
  v50 = *(_BYTE *)(v15 + 16);
  if ( v50 <= 0x17u )
  {
    v48 = 0;
    if ( v50 == 5 && *(_WORD *)(v15 + 18) == 32 )
    {
LABEL_79:
      v48 = 0;
      if ( (*(_BYTE *)(v15 + 17) & 2) != 0 )
      {
        v51 = (_QWORD *)sub_13CF970(v15);
        v48 = 0;
        if ( v23 == *v51 )
        {
          v52 = sub_146F1B0(a1, v23);
          v53 = sub_146F1B0(a1, v15);
          v54 = sub_14806B0(a1, v53, v52, 0, 0);
          v48 = (unsigned __int8)sub_1477C30(a1, v54) == 0 ? 1 : 3;
        }
      }
    }
  }
  else
  {
    v48 = 0;
    if ( v50 == 56 )
      goto LABEL_79;
  }
LABEL_73:
  v64 = v48;
  v49 = sub_146F1B0(a1, v61);
  v21 = sub_14799E0(a1, v49, (__int64)v39, v59, v64);
  sub_1464870(a1, v23, v57);
  sub_1457D90(&v69, v23, a1);
  sub_1467C40(v56, &v69)[5] = v21;
  v69 = &unk_49EE2B0;
  sub_1455FA0((__int64)&v70);
  if ( *(_BYTE *)(v15 + 16) > 0x17u && sub_146CEE0(a1, (__int64)v39, v59) && (unsigned __int8)sub_1471300(a1, v15, v59) )
  {
    v55 = sub_13A5B00(a1, v49, (__int64)v39, 0, 0);
    sub_14799E0(a1, v55, (__int64)v39, v59, v64);
  }
  if ( v77 != &v79 )
    _libc_free((unsigned __int64)v77);
  return v21;
}
