// Function: sub_1A129E0
// Address: 0x1a129e0
//
__int64 __fastcall sub_1A129E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // r10
  __int64 v13; // r13
  __int64 v14; // rcx
  unsigned int v15; // edx
  unsigned int *v16; // rsi
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // edx
  unsigned int *v21; // rsi
  __int64 v22; // r10
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // r13
  _QWORD *v26; // rax
  __int64 v27; // r8
  int v28; // r9d
  int v29; // r10d
  char v30; // al
  int v31; // r10d
  __int64 v32; // r11
  unsigned int v33; // eax
  char v34; // al
  __int64 v35; // r11
  char v36; // r13
  __int64 v37; // rdx
  int v38; // esi
  __int64 *v39; // rax
  int v40; // esi
  __int64 *v41; // rax
  int v42; // r11d
  __int64 *v43; // r10
  int v44; // eax
  int v45; // edx
  int v46; // r11d
  int v47; // r8d
  int v48; // eax
  int v49; // ecx
  __int64 v50; // rdi
  unsigned int v51; // eax
  __int64 v52; // rsi
  int v53; // r11d
  __int64 *v54; // r10
  int v55; // eax
  int v56; // eax
  __int64 v57; // rsi
  int v58; // r10d
  unsigned int v59; // r13d
  __int64 *v60; // rdi
  __int64 v61; // rcx
  char v62; // [rsp+Fh] [rbp-D1h]
  char v63; // [rsp+Fh] [rbp-D1h]
  int v64; // [rsp+20h] [rbp-C0h]
  __int64 **v65; // [rsp+20h] [rbp-C0h]
  __int64 v66; // [rsp+20h] [rbp-C0h]
  __int64 **v67; // [rsp+28h] [rbp-B8h]
  int v68; // [rsp+28h] [rbp-B8h]
  int v69; // [rsp+28h] [rbp-B8h]
  __int64 v70; // [rsp+28h] [rbp-B8h]
  __int64 v71; // [rsp+28h] [rbp-B8h]
  __int64 v72; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v73; // [rsp+38h] [rbp-A8h]
  __int64 v74; // [rsp+40h] [rbp-A0h]
  unsigned int v75; // [rsp+48h] [rbp-98h]
  int v76; // [rsp+50h] [rbp-90h] BYREF
  _QWORD *v77; // [rsp+58h] [rbp-88h] BYREF
  unsigned int v78; // [rsp+60h] [rbp-80h]
  __int64 v79; // [rsp+68h] [rbp-78h]
  unsigned int v80; // [rsp+70h] [rbp-70h]
  unsigned int v81; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v82; // [rsp+88h] [rbp-58h] BYREF
  unsigned int v83; // [rsp+90h] [rbp-50h]
  __int64 v84; // [rsp+98h] [rbp-48h]
  unsigned int v85; // [rsp+A0h] [rbp-40h]

  v4 = a1 + 120;
  v5 = *(_DWORD *)(a1 + 144);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_81;
  }
  v6 = *(_QWORD *)(a1 + 128);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
LABEL_3:
    result = v8[1] ^ 6;
    if ( (result & 6) == 0 )
      return result;
    goto LABEL_4;
  }
  v42 = 1;
  v43 = 0;
  while ( v9 != -8 )
  {
    if ( !v43 && v9 == -16 )
      v43 = v8;
    v7 = (v5 - 1) & (v42 + v7);
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      goto LABEL_3;
    ++v42;
  }
  v44 = *(_DWORD *)(a1 + 136);
  if ( v43 )
    v8 = v43;
  ++*(_QWORD *)(a1 + 120);
  v45 = v44 + 1;
  if ( 4 * (v44 + 1) >= 3 * v5 )
  {
LABEL_81:
    sub_1A0FE70(v4, 2 * v5);
    v48 = *(_DWORD *)(a1 + 144);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(a1 + 128);
      v51 = (v48 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v45 = *(_DWORD *)(a1 + 136) + 1;
      v8 = (__int64 *)(v50 + 16LL * v51);
      v52 = *v8;
      if ( a2 != *v8 )
      {
        v53 = 1;
        v54 = 0;
        while ( v52 != -8 )
        {
          if ( !v54 && v52 == -16 )
            v54 = v8;
          v51 = v49 & (v53 + v51);
          v8 = (__int64 *)(v50 + 16LL * v51);
          v52 = *v8;
          if ( a2 == *v8 )
            goto LABEL_71;
          ++v53;
        }
        if ( v54 )
          v8 = v54;
      }
      goto LABEL_71;
    }
    goto LABEL_109;
  }
  if ( v5 - *(_DWORD *)(a1 + 140) - v45 <= v5 >> 3 )
  {
    sub_1A0FE70(v4, v5);
    v55 = *(_DWORD *)(a1 + 144);
    if ( v55 )
    {
      v56 = v55 - 1;
      v57 = *(_QWORD *)(a1 + 128);
      v58 = 1;
      v59 = v56 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v45 = *(_DWORD *)(a1 + 136) + 1;
      v60 = 0;
      v8 = (__int64 *)(v57 + 16LL * v59);
      v61 = *v8;
      if ( a2 != *v8 )
      {
        while ( v61 != -8 )
        {
          if ( !v60 && v61 == -16 )
            v60 = v8;
          v59 = v56 & (v58 + v59);
          v8 = (__int64 *)(v57 + 16LL * v59);
          v61 = *v8;
          if ( a2 == *v8 )
            goto LABEL_71;
          ++v58;
        }
        if ( v60 )
          v8 = v60;
      }
      goto LABEL_71;
    }
LABEL_109:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
LABEL_71:
  *(_DWORD *)(a1 + 136) = v45;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 140);
  *v8 = a2;
  v8[1] = 0;
LABEL_4:
  v11 = *(unsigned int *)(a1 + 176);
  v12 = *(_QWORD *)(a2 - 48);
  v13 = *(_QWORD *)(a2 - 24);
  if ( !(_DWORD)v11 )
    goto LABEL_58;
  v14 = *(_QWORD *)(a1 + 160);
  v15 = (v11 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v16 = (unsigned int *)(v14 + 48LL * v15);
  v17 = *(_QWORD *)v16;
  if ( v12 != *(_QWORD *)v16 )
  {
    v40 = 1;
    while ( v17 != -8 )
    {
      v47 = v40 + 1;
      v15 = (v11 - 1) & (v40 + v15);
      v16 = (unsigned int *)(v14 + 48LL * v15);
      v17 = *(_QWORD *)v16;
      if ( v12 == *(_QWORD *)v16 )
        goto LABEL_6;
      v40 = v47;
    }
    goto LABEL_58;
  }
LABEL_6:
  if ( v16 == (unsigned int *)(v14 + 48 * v11) )
  {
LABEL_58:
    v41 = sub_1A10F60(a1, *(_QWORD *)(a2 - 48));
    sub_1A111D0(&v76, v41);
    goto LABEL_8;
  }
  v76 = 0;
  sub_1A0F8E0(&v76, v16 + 2);
LABEL_8:
  v18 = *(unsigned int *)(a1 + 176);
  if ( !(_DWORD)v18 )
    goto LABEL_55;
  v19 = *(_QWORD *)(a1 + 160);
  v20 = (v18 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v21 = (unsigned int *)(v19 + 48LL * v20);
  v22 = *(_QWORD *)v21;
  if ( v13 != *(_QWORD *)v21 )
  {
    v38 = 1;
    while ( v22 != -8 )
    {
      v46 = v38 + 1;
      v20 = (v18 - 1) & (v38 + v20);
      v21 = (unsigned int *)(v19 + 48LL * v20);
      v22 = *(_QWORD *)v21;
      if ( v13 == *(_QWORD *)v21 )
        goto LABEL_10;
      v38 = v46;
    }
    goto LABEL_55;
  }
LABEL_10:
  if ( v21 == (unsigned int *)(v19 + 48 * v18) )
  {
LABEL_55:
    v39 = sub_1A10F60(a1, v13);
    sub_1A111D0((int *)&v81, v39);
    goto LABEL_12;
  }
  v81 = 0;
  sub_1A0F8E0((int *)&v81, v21 + 2);
LABEL_12:
  v23 = v76;
  if ( !v76 || !v81 )
  {
    v24 = sub_1599EF0(*(__int64 ***)a2);
    goto LABEL_15;
  }
  v29 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( v81 == 1 && v76 == 1 )
  {
    v24 = sub_15A37B0(v29, v77, v82, 0);
    goto LABEL_15;
  }
  v67 = *(__int64 ***)a2;
  if ( v76 == 3 && v81 == 3 )
  {
    v64 = *(_WORD *)(a2 + 18) & 0x7FFF;
    sub_1590F80((__int64)&v72, v29, (__int64)&v82);
    v30 = sub_158BB40((__int64)&v72, (__int64)&v77);
    v31 = v64;
    v32 = (__int64)v67;
    if ( v75 > 0x40 && v74 )
    {
      v62 = v30;
      v65 = v67;
      v68 = v31;
      j_j___libc_free_0_0(v74);
      v30 = v62;
      v32 = (__int64)v65;
      v31 = v68;
    }
    if ( v73 > 0x40 && v72 )
    {
      v63 = v30;
      v66 = v32;
      v69 = v31;
      j_j___libc_free_0_0(v72);
      v30 = v63;
      v32 = v66;
      v31 = v69;
    }
    if ( v30 )
    {
      v24 = sub_15A0600(v32);
    }
    else
    {
      v70 = v32;
      v33 = sub_15FF0F0(v31);
      sub_1590F80((__int64)&v72, v33, (__int64)&v82);
      v34 = sub_158BB40((__int64)&v72, (__int64)&v77);
      v35 = v70;
      v36 = v34;
      if ( v75 > 0x40 && v74 )
      {
        j_j___libc_free_0_0(v74);
        v35 = v70;
      }
      if ( v73 > 0x40 && v72 )
      {
        v71 = v35;
        j_j___libc_free_0_0(v72);
        v35 = v71;
      }
      if ( !v36 )
      {
LABEL_41:
        v23 = v76;
        goto LABEL_42;
      }
      v24 = sub_15A0640(v35);
    }
LABEL_15:
    if ( v24 )
    {
      if ( *(_BYTE *)(v24 + 16) != 9 )
      {
        v72 = a2;
        v25 = v24 | 2;
        v26 = sub_1A10690(v4, &v72);
        sub_1A10580(a1, v26 + 1, a2, v25, v27, v28);
      }
      result = v81;
LABEL_19:
      if ( (_DWORD)result != 3 )
        goto LABEL_20;
LABEL_47:
      if ( v85 > 0x40 && v84 )
        result = j_j___libc_free_0_0(v84);
      if ( v83 > 0x40 && v82 )
        result = j_j___libc_free_0_0(v82);
      goto LABEL_20;
    }
    goto LABEL_41;
  }
LABEL_42:
  if ( v23 != 4 )
  {
    result = v81;
    if ( v81 != 4 )
    {
      v37 = (v8[1] >> 1) & 3;
      if ( v37 != 1 && v37 != 2 )
        goto LABEL_19;
    }
  }
  result = sub_1A11830(a1, a2);
  if ( v81 == 3 )
    goto LABEL_47;
LABEL_20:
  if ( v76 == 3 )
  {
    if ( v80 > 0x40 && v79 )
      result = j_j___libc_free_0_0(v79);
    if ( v78 > 0x40 )
    {
      if ( v77 )
        return j_j___libc_free_0_0(v77);
    }
  }
  return result;
}
