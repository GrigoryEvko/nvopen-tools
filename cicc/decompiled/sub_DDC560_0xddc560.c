// Function: sub_DDC560
// Address: 0xddc560
//
__int64 __fastcall sub_DDC560(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rcx
  int v10; // eax
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 *v17; // r9
  __int64 v18; // r12
  __int64 i; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // r12
  __int64 v24; // r14
  __int64 result; // rax
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  char *v32; // r13
  void (__fastcall *v33)(__m128i **, const __m128i **, int); // rax
  char *v34; // r14
  __int64 v35; // rdx
  int v36; // eax
  int v37; // r9d
  __int64 v38; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+8h] [rbp-C8h] BYREF
  __int64 v40; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+18h] [rbp-B8h] BYREF
  bool v42; // [rsp+2Dh] [rbp-A3h] BYREF
  char v43; // [rsp+2Eh] [rbp-A2h] BYREF
  char v44; // [rsp+2Fh] [rbp-A1h] BYREF
  int v45; // [rsp+30h] [rbp-A0h] BYREF
  char v46; // [rsp+34h] [rbp-9Ch]
  __int64 v47; // [rsp+38h] [rbp-98h] BYREF
  char *v48; // [rsp+40h] [rbp-90h] BYREF
  __int64 *v49; // [rsp+48h] [rbp-88h]
  char *v50; // [rsp+50h] [rbp-80h]
  __int64 **v51[2]; // [rsp+60h] [rbp-70h] BYREF
  __int64 (__fastcall *v52)(__m128i **, const __m128i **, int); // [rsp+70h] [rbp-60h]
  __int64 (__fastcall *v53)(__int64 ***, __int64 *); // [rsp+78h] [rbp-58h]
  __int64 *v54; // [rsp+80h] [rbp-50h]
  bool *v55; // [rsp+88h] [rbp-48h]
  char **v56; // [rsp+90h] [rbp-40h]

  v40 = a3;
  v5 = a1[5];
  v41 = a2;
  v39 = a4;
  v38 = a5;
  if ( a2 )
  {
    v6 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v7 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v6 = 0;
    v7 = 0;
  }
  if ( v7 >= *(_DWORD *)(v5 + 32) || !*(_QWORD *)(*(_QWORD *)(v5 + 24) + 8 * v6) )
    return 1;
  v46 = BYTE4(v40);
  v48 = &v43;
  v49 = (__int64 *)&v45;
  v45 = sub_B531B0(v40);
  v43 = 0;
  v44 = 0;
  v50 = &v44;
  v42 = v45 != (_DWORD)v40;
  if ( v45 != (_DWORD)v40 )
  {
    v52 = 0;
    v31 = sub_22077B0(24);
    if ( v31 )
    {
      *(_QWORD *)(v31 + 16) = a1;
      *(_QWORD *)v31 = &v39;
      *(_QWORD *)(v31 + 8) = &v38;
    }
    v51[0] = (__int64 **)v31;
    v32 = v48;
    v53 = sub_DCFA30;
    v33 = (void (__fastcall *)(__m128i **, const __m128i **, int))sub_D913F0;
    v52 = sub_D913F0;
    if ( *v48 )
    {
      v34 = v50;
      if ( *v50 )
      {
        if ( !*v48 )
        {
LABEL_47:
          a2 = (__int64)v51;
          v33((__m128i **)v51, (const __m128i **)v51, 3);
          goto LABEL_6;
        }
        goto LABEL_57;
      }
      LODWORD(v47) = 33;
      BYTE4(v47) = 0;
    }
    else
    {
      a2 = (__int64)&v47;
      v47 = *v49;
      *v32 = sub_DCFA30(v51, &v47);
      v34 = v50;
      v33 = (void (__fastcall *)(__m128i **, const __m128i **, int))v52;
      if ( *v50 )
      {
        if ( *v48 )
          goto LABEL_56;
        goto LABEL_51;
      }
      LODWORD(v47) = 33;
      BYTE4(v47) = 0;
      if ( !v52 )
        sub_4263D6(v51, &v47, v35);
    }
    a2 = (__int64)&v47;
    *v34 = v53(v51, &v47);
    v33 = (void (__fastcall *)(__m128i **, const __m128i **, int))v52;
    if ( *v48 && *v50 )
    {
LABEL_56:
      if ( !v33 )
        return 1;
LABEL_57:
      v33((__m128i **)v51, (const __m128i **)v51, 3);
      return 1;
    }
LABEL_51:
    if ( !v33 )
      goto LABEL_6;
    goto LABEL_47;
  }
LABEL_6:
  v54 = a1;
  v51[0] = (__int64 **)&v41;
  v51[1] = (__int64 **)&v40;
  v52 = (__int64 (__fastcall *)(__m128i **, const __m128i **, int))&v39;
  v53 = (__int64 (__fastcall *)(__int64 ***, __int64 *))&v38;
  v55 = &v42;
  v56 = &v48;
  v8 = a1[6];
  v9 = *(_QWORD *)(v8 + 8);
  v10 = *(_DWORD *)(v8 + 24);
  if ( !v10 )
    goto LABEL_10;
  a2 = (unsigned int)(v10 - 1);
  v11 = a2 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v41 != *v12 )
  {
    v36 = 1;
    while ( v13 != -4096 )
    {
      v37 = v36 + 1;
      v11 = a2 & (v36 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v41 == *v12 )
        goto LABEL_8;
      v36 = v37;
    }
    goto LABEL_10;
  }
LABEL_8:
  v14 = v12[1];
  if ( !v14 || v41 != **(_QWORD **)(v14 + 32) )
  {
LABEL_10:
    v18 = sub_AA54C0(v41);
    goto LABEL_11;
  }
  v18 = sub_D47840(v14);
LABEL_11:
  for ( i = v41; v18; v18 = sub_D98160((__int64)a1, v18) )
  {
    v20 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v20 == v18 + 48 || !v20 || (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 > 0xA )
      BUG();
    if ( *(_BYTE *)(v20 - 24) == 31
      && (*(_DWORD *)(v20 - 20) & 0x7FFFFFF) != 1
      && (unsigned __int8)sub_DDC390((__int64 *)v51, *(_QWORD *)(v20 - 120), *(_QWORD *)(v20 - 56) != i) )
    {
      return 1;
    }
    a2 = v18;
  }
  v21 = a1[4];
  if ( !*(_BYTE *)(v21 + 192) )
    sub_CFDFC0(a1[4], a2, i, v15, v16, v17);
  v22 = *(_QWORD *)(v21 + 16);
  v23 = v22 + 32LL * *(unsigned int *)(v21 + 24);
  if ( v23 == v22 )
  {
LABEL_26:
    v26 = sub_B6AC80(*(_QWORD *)(*a1 + 40), 153);
    if ( !v26 )
      return 0;
    v27 = *(_QWORD *)(v26 + 16);
    if ( !v27 )
      return 0;
    while ( 1 )
    {
      v28 = *(_QWORD *)(v27 + 24);
      if ( *(_BYTE *)v28 == 85 )
      {
        v29 = *(_QWORD *)(v28 - 32);
        if ( v29 )
        {
          if ( !*(_BYTE *)v29 && *(_QWORD *)(v29 + 24) == *(_QWORD *)(v28 + 80) && (*(_BYTE *)(v29 + 33) & 0x20) != 0 )
          {
            v30 = sub_B43CB0(*(_QWORD *)(v27 + 24));
            if ( *(_QWORD *)(v41 + 72) == v30
              && (unsigned __int8)sub_B19D00(a1[5], v28, v41)
              && (unsigned __int8)sub_DDC390(
                                    (__int64 *)v51,
                                    *(_QWORD *)(v28 - 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF)),
                                    0) )
            {
              break;
            }
          }
        }
      }
      v27 = *(_QWORD *)(v27 + 8);
      if ( !v27 )
        return 0;
    }
    return 1;
  }
  while ( 1 )
  {
    v24 = *(_QWORD *)(v22 + 16);
    if ( v24 )
    {
      if ( (unsigned __int8)sub_B19D00(a1[5], *(_QWORD *)(v22 + 16), v41) )
      {
        result = sub_DDC390((__int64 *)v51, *(_QWORD *)(v24 - 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF)), 0);
        if ( (_BYTE)result )
          return result;
      }
    }
    v22 += 32;
    if ( v22 == v23 )
      goto LABEL_26;
  }
}
