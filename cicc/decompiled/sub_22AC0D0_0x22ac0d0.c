// Function: sub_22AC0D0
// Address: 0x22ac0d0
//
__int64 __fastcall sub_22AC0D0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // rax
  unsigned __int8 **i; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r14d
  __int64 v9; // r12
  unsigned __int8 **v10; // rax
  char v12; // dl
  unsigned __int8 *v13; // rdi
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned __int8 *v16; // rax
  unsigned __int8 *v17; // rdi
  unsigned __int8 *v18; // rsi
  unsigned __int8 *v19; // rsi
  unsigned __int8 **v20; // rax
  unsigned __int8 **v21; // rdx
  __int64 *v22; // r12
  char *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r15
  __int64 v27; // rcx
  _BYTE *v28; // rsi
  char *v29; // rax
  _BYTE *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rdi
  int v33; // ecx
  __int64 v34; // r9
  int v35; // ecx
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // r11
  __int64 v39; // rax
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rcx
  unsigned __int64 v44; // rax
  __int64 *v45; // rcx
  __int64 v46; // r9
  unsigned __int64 v47; // rax
  char *v48; // rax
  __int64 *v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  int v52; // eax
  int v53; // r10d
  unsigned __int64 v54; // rbx
  unsigned __int64 *v55; // rcx
  unsigned __int64 v56; // rdx
  bool v57; // zf
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v61; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v62; // [rsp+18h] [rbp-A8h] BYREF
  _BYTE *v63; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v64[4]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v65; // [rsp+50h] [rbp-70h] BYREF
  char *v66; // [rsp+58h] [rbp-68h]
  __int64 v67; // [rsp+60h] [rbp-60h]
  int v68; // [rsp+68h] [rbp-58h]
  unsigned __int8 v69; // [rsp+6Ch] [rbp-54h]
  char v70; // [rsp+70h] [rbp-50h] BYREF

  v62 = a2;
  v3 = sub_B43CC0((__int64)a2);
  v8 = *(unsigned __int8 *)(a1 + 68);
  v9 = v3;
  if ( !(_BYTE)v8 )
    goto LABEL_8;
  v10 = *(unsigned __int8 ***)(a1 + 48);
  v5 = *(unsigned int *)(a1 + 60);
  for ( i = &v10[v5]; i != v10; ++v10 )
  {
    if ( v62 == *v10 )
      return v8;
  }
  if ( (unsigned int)v5 < *(_DWORD *)(a1 + 56) )
  {
    *(_DWORD *)(a1 + 60) = v5 + 1;
    *i = a2;
    ++*(_QWORD *)(a1 + 40);
  }
  else
  {
LABEL_8:
    v8 = 1;
    sub_C8CC70(a1 + 40, (__int64)a2, (__int64)i, v5, v6, v7);
    if ( !v12 )
      return v8;
  }
  if ( !sub_D97040(*(_QWORD *)(a1 + 32), *((_QWORD *)v62 + 1)) )
    return 0;
  v13 = v62;
  if ( *v62 != 84 )
  {
    if ( !sub_991A70(v62, 0, 0, 0, 0, 1u, 0) )
      return 0;
    v13 = v62;
  }
  v14 = sub_D97050(*(_QWORD *)(a1 + 32), *((_QWORD *)v13 + 1));
  if ( v14 > 0x40 )
    return 0;
  v15 = *(_QWORD *)(v9 + 40);
  v16 = *(unsigned __int8 **)(v9 + 32);
  v17 = &v16[v15];
  if ( v15 >> 2 <= 0 )
    goto LABEL_61;
  v18 = &v16[4 * (v15 >> 2)];
  do
  {
    if ( v14 == *v16 )
      goto LABEL_21;
    if ( v14 == v16[1] )
    {
      ++v16;
      goto LABEL_21;
    }
    if ( v14 == v16[2] )
    {
      v16 += 2;
      goto LABEL_21;
    }
    if ( v14 == v16[3] )
    {
      v16 += 3;
      goto LABEL_21;
    }
    v16 += 4;
  }
  while ( v16 != v18 );
  v15 = v17 - v16;
LABEL_61:
  if ( v15 == 2 )
  {
LABEL_87:
    if ( v14 != *v16 )
    {
      ++v16;
      goto LABEL_64;
    }
    goto LABEL_21;
  }
  if ( v15 == 3 )
  {
    if ( v14 == *v16 )
      goto LABEL_21;
    ++v16;
    goto LABEL_87;
  }
  if ( v15 != 1 )
    return 0;
LABEL_64:
  v8 = 0;
  if ( v14 != *v16 )
    return v8;
LABEL_21:
  if ( v17 == v16 )
    return 0;
  v19 = v62;
  if ( !*(_BYTE *)(a1 + 244) )
  {
    if ( !sub_C8CA60(a1 + 216, (__int64)v62) )
    {
      v19 = v62;
      goto LABEL_31;
    }
    return 0;
  }
  v20 = *(unsigned __int8 ***)(a1 + 224);
  v21 = &v20[*(unsigned int *)(a1 + 236)];
  if ( v20 != v21 )
  {
    while ( v62 != *v20 )
    {
      if ( v21 == ++v20 )
        goto LABEL_31;
    }
    return 0;
  }
LABEL_31:
  v22 = sub_DD8400(*(_QWORD *)(a1 + 32), (__int64)v19);
  v8 = sub_22ABC20((__int64)v22, (__int64)v62, *(_QWORD *)a1, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 16));
  if ( !(_BYTE)v8 )
    return 0;
  v67 = 4;
  v66 = &v70;
  v68 = 0;
  v69 = 1;
  v26 = *((_QWORD *)v62 + 2);
  v65 = 0;
  if ( !v26 )
    return v8;
  v27 = v8;
  while ( 1 )
  {
    v28 = *(_BYTE **)(v26 + 24);
    v63 = v28;
    if ( (_BYTE)v27 )
    {
      v29 = v66;
      v23 = &v66[8 * HIDWORD(v67)];
      if ( v66 != v23 )
      {
        while ( v28 != *(_BYTE **)v29 )
        {
          v29 += 8;
          if ( v23 == v29 )
            goto LABEL_58;
        }
        goto LABEL_39;
      }
LABEL_58:
      if ( HIDWORD(v67) < (unsigned int)v67 )
        break;
    }
    sub_C8CC70((__int64)&v65, (__int64)v28, (__int64)v23, v27, v24, v25);
    v27 = v69;
    if ( (_BYTE)v23 )
      goto LABEL_43;
LABEL_39:
    v26 = *(_QWORD *)(v26 + 8);
    if ( !v26 )
      goto LABEL_40;
  }
  ++HIDWORD(v67);
  *(_QWORD *)v23 = v28;
  v27 = v69;
  ++v65;
LABEL_43:
  v30 = v63;
  if ( *v63 == 84 )
  {
    if ( *(_BYTE *)(a1 + 68) )
    {
      v48 = *(char **)(a1 + 48);
      v23 = &v48[8 * *(unsigned int *)(a1 + 60)];
      if ( v48 != v23 )
      {
        while ( v63 != *(_BYTE **)v48 )
        {
          v48 += 8;
          if ( v23 == v48 )
            goto LABEL_44;
        }
        goto LABEL_39;
      }
    }
    else
    {
      if ( sub_C8CA60(a1 + 40, (__int64)v63) )
        goto LABEL_57;
      v30 = v63;
    }
  }
LABEL_44:
  v31 = *(_QWORD *)(a1 + 16);
  v32 = *((_QWORD *)v30 + 5);
  v33 = *(_DWORD *)(v31 + 24);
  v34 = *(_QWORD *)(v31 + 8);
  if ( v33 )
  {
    v35 = v33 - 1;
    v36 = v35 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v37 = (__int64 *)(v34 + 16LL * v36);
    v38 = *v37;
    if ( v32 == *v37 )
    {
LABEL_46:
      v39 = v37[1];
      goto LABEL_47;
    }
    v52 = 1;
    while ( v38 != -4096 )
    {
      v53 = v52 + 1;
      v36 = v35 & (v52 + v36);
      v37 = (__int64 *)(v34 + 16LL * v36);
      v38 = *v37;
      if ( v32 == *v37 )
        goto LABEL_46;
      v52 = v53;
    }
  }
  v39 = 0;
LABEL_47:
  if ( *(_QWORD *)a1 != v39 )
  {
    if ( *v30 == 84 )
      goto LABEL_54;
    if ( *(_BYTE *)(a1 + 68) )
    {
      v40 = *(_QWORD **)(a1 + 48);
      v41 = &v40[*(unsigned int *)(a1 + 60)];
      if ( v40 != v41 )
      {
        while ( v30 != (_BYTE *)*v40 )
        {
          if ( v41 == ++v40 )
            goto LABEL_77;
        }
        goto LABEL_54;
      }
      goto LABEL_77;
    }
    goto LABEL_76;
  }
  if ( !*(_BYTE *)(a1 + 68) )
  {
LABEL_76:
    v49 = sub_C8CA60(a1 + 40, (__int64)v30);
    v30 = v63;
    if ( v49 )
      goto LABEL_54;
    goto LABEL_77;
  }
  v50 = *(_QWORD **)(a1 + 48);
  v51 = &v50[*(unsigned int *)(a1 + 60)];
  if ( v50 != v51 )
  {
    while ( v30 != (_BYTE *)*v50 )
    {
      if ( v51 == ++v50 )
        goto LABEL_77;
    }
    goto LABEL_54;
  }
LABEL_77:
  if ( (unsigned __int8)sub_22AC0D0(a1, v30) )
    goto LABEL_57;
  v30 = v63;
LABEL_54:
  v42 = sub_22ABF70(a1, (__int64)v30, (__int64)v62);
  v43 = *(_QWORD *)(a1 + 32);
  v64[2] = a1;
  v64[3] = v42;
  v64[0] = &v63;
  v64[1] = &v62;
  v60 = v42;
  v44 = sub_1055A10((unsigned __int64)v22, (__int64)sub_22ABA70, (__int64)v64, v43);
  v45 = (__int64 *)v44;
  if ( (__int64 *)v44 == v22
    || (v46 = v60,
        v61 = v44,
        v47 = sub_1055AA0(v44, v46 + 80, *(_QWORD *)(a1 + 32)),
        v45 = (__int64 *)v61,
        v22 == (__int64 *)v47) )
  {
    v22 = v45;
LABEL_57:
    v27 = v69;
    goto LABEL_39;
  }
  v54 = *(_QWORD *)(a1 + 200) & 0xFFFFFFFFFFFFFFF8LL;
  v55 = *(unsigned __int64 **)(v54 + 8);
  v56 = *(_QWORD *)v54 & 0xFFFFFFFFFFFFFFF8LL;
  *v55 = v56 | *v55 & 7;
  *(_QWORD *)(v56 + 8) = v55;
  *(_QWORD *)v54 &= 7uLL;
  v57 = *(_BYTE *)(v54 + 76) == 0;
  *(_QWORD *)(v54 + 8) = 0;
  *(_QWORD *)(v54 - 32) = &unk_4A09CC0;
  if ( v57 )
    _libc_free(*(_QWORD *)(v54 + 56));
  v58 = *(_QWORD *)(v54 + 40);
  if ( v58 != 0 && v58 != -4096 && v58 != -8192 )
    sub_BD60C0((_QWORD *)(v54 + 24));
  *(_QWORD *)(v54 - 32) = &unk_49DB368;
  v59 = *(_QWORD *)(v54 - 8);
  if ( v59 != 0 && v59 != -4096 && v59 != -8192 )
    sub_BD60C0((_QWORD *)(v54 - 24));
  v8 = 0;
  j_j___libc_free_0(v54 - 32);
  LOBYTE(v27) = v69;
LABEL_40:
  if ( !(_BYTE)v27 )
    _libc_free((unsigned __int64)v66);
  return v8;
}
