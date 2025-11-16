// Function: sub_1940670
// Address: 0x1940670
//
__int64 __fastcall sub_1940670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, __m128i a6, __m128i a7)
{
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int8 *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r12
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // r8
  char v34; // di
  unsigned int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rbx
  _BYTE *v41; // r13
  __int64 v42; // rax
  unsigned __int64 v43; // rbx
  unsigned __int8 *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // [rsp+0h] [rbp-B0h]
  __int64 v48; // [rsp+0h] [rbp-B0h]
  __int64 v49; // [rsp+0h] [rbp-B0h]
  __int64 *v50; // [rsp+8h] [rbp-A8h]
  __int64 v51; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v52; // [rsp+8h] [rbp-A8h]
  __int64 v53; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v54[2]; // [rsp+10h] [rbp-A0h] BYREF
  char v55; // [rsp+20h] [rbp-90h]
  char v56; // [rsp+21h] [rbp-8Fh]
  unsigned __int8 *v57; // [rsp+30h] [rbp-80h] BYREF
  __int64 v58; // [rsp+38h] [rbp-78h]
  __int64 v59; // [rsp+40h] [rbp-70h] BYREF
  __int64 v60; // [rsp+48h] [rbp-68h]
  __int64 v61; // [rsp+50h] [rbp-60h]
  int v62; // [rsp+58h] [rbp-58h]
  __int64 v63; // [rsp+60h] [rbp-50h]
  __int64 v64; // [rsp+68h] [rbp-48h]

  v12 = sub_146F1B0((__int64)a5, a1);
  if ( *(_WORD *)(v12 + 24) != 7 )
    BUG();
  v13 = v12;
  v14 = **(_QWORD **)(v12 + 32);
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 15 )
  {
    v51 = **(_QWORD **)(v12 + 32);
    v48 = v12;
    if ( *(_BYTE *)(sub_1456040(a2) + 8) != 15 )
    {
      v28 = sub_1456040(v51);
      v53 = sub_1456E10((__int64)a5, v28);
      v29 = sub_1483B20(a5, a2, v53, a6, a7);
      v30 = sub_13F9E70(a3);
      v31 = sub_157EBA0(v30);
      v32 = sub_38767A0(a4, v29, v53, v31);
      v33 = sub_13FC520(a3);
      v34 = *(_BYTE *)(a1 + 23) & 0x40;
      v35 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      if ( v35 )
      {
        v36 = 24LL * *(unsigned int *)(a1 + 56) + 8;
        v37 = 0;
        while ( 1 )
        {
          v38 = a1 - 24LL * v35;
          if ( v34 )
            v38 = *(_QWORD *)(a1 - 8);
          if ( v33 == *(_QWORD *)(v38 + v36) )
            break;
          ++v37;
          v36 += 8;
          if ( v35 == (_DWORD)v37 )
            goto LABEL_37;
        }
        v39 = 24 * v37;
      }
      else
      {
LABEL_37:
        v39 = 0x17FFFFFFE8LL;
      }
      if ( v34 )
        v40 = *(_QWORD *)(a1 - 8);
      else
        v40 = a1 - 24LL * v35;
      v41 = *(_BYTE **)(v40 + v39);
      v42 = sub_13FC520(a3);
      v43 = sub_157EBA0(v42);
      v57 = 0;
      v60 = sub_16498A0(v43);
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      v58 = *(_QWORD *)(v43 + 40);
      v59 = v43 + 24;
      v44 = *(unsigned __int8 **)(v43 + 48);
      v54[0] = v44;
      if ( v44 )
      {
        sub_1623A60((__int64)v54, (__int64)v44, 2);
        if ( v57 )
          sub_161E7C0((__int64)&v57, (__int64)v57);
        v57 = v54[0];
        if ( v54[0] )
          sub_1623210((__int64)v54, v54[0], (__int64)&v57);
      }
      v56 = 1;
      v54[0] = "lftr.limit";
      v55 = 3;
      v45 = sub_12815B0((__int64 *)&v57, 0, v41, v32, (__int64)v54);
      v21 = (__int64)v57;
      v22 = v45;
      if ( v57 )
        goto LABEL_12;
      return v22;
    }
    v13 = v48;
    v14 = **(_QWORD **)(v48 + 32);
  }
  v47 = v13;
  v50 = (__int64 *)a2;
  if ( !sub_14560B0(v14) )
  {
    v49 = **(_QWORD **)(v47 + 32);
    v24 = sub_1456040(v49);
    v52 = sub_1456C90((__int64)a5, v24);
    v25 = sub_1456040(a2);
    v26 = sub_1456C90((__int64)a5, v25);
    v27 = v49;
    if ( v52 > v26 )
    {
      v46 = sub_1456040(a2);
      v27 = sub_14835F0(a5, v49, v46, 0, a6, a7);
    }
    v57 = (unsigned __int8 *)&v59;
    v59 = v27;
    v60 = a2;
    v58 = 0x200000002LL;
    v50 = sub_147DD40((__int64)a5, (__int64 *)&v57, 0, 0, a6, a7);
    if ( v57 != (unsigned __int8 *)&v59 )
      _libc_free((unsigned __int64)v57);
  }
  v15 = sub_13F9E70(a3);
  v16 = sub_157EBA0(v15);
  v17 = sub_16498A0(v16);
  v57 = 0;
  v60 = v17;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v58 = *(_QWORD *)(v16 + 40);
  v59 = v16 + 24;
  v18 = *(unsigned __int8 **)(v16 + 48);
  v54[0] = v18;
  if ( v18 )
  {
    sub_1623A60((__int64)v54, (__int64)v18, 2);
    if ( v57 )
      sub_161E7C0((__int64)&v57, (__int64)v57);
    v57 = v54[0];
    if ( v54[0] )
      sub_1623210((__int64)v54, v54[0], (__int64)&v57);
  }
  if ( *(_BYTE *)(sub_1456040(a2) + 8) == 15 )
    v19 = *(_QWORD *)a1;
  else
    v19 = sub_1456040(a2);
  v20 = sub_38767A0(a4, v50, v19, v16);
  v21 = (__int64)v57;
  v22 = v20;
  if ( v57 )
LABEL_12:
    sub_161E7C0((__int64)&v57, v21);
  return v22;
}
