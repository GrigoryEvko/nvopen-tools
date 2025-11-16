// Function: sub_35AE230
// Address: 0x35ae230
//
void __fastcall sub_35AE230(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // edx
  __int64 v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // rdi
  _QWORD *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 *v18; // rbx
  __int64 *v19; // r14
  __int64 v20; // r15
  __int64 *v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 v24; // r15
  __int64 v25; // r12
  __int64 *v26; // rax
  __int64 *v27; // r14
  __int64 *v28; // r13
  __int64 v29; // r8
  __int64 *v30; // r12
  __int64 v31; // r14
  __int64 *v32; // rax
  _QWORD *v33; // r13
  __int64 v34; // r15
  __int64 *v35; // rax
  __int64 *v36; // rdx
  char v37; // al
  int v38; // r14d
  __int64 v40; // [rsp+18h] [rbp-118h]
  __int64 v41; // [rsp+20h] [rbp-110h]
  __int64 v42; // [rsp+20h] [rbp-110h]
  __int64 v43; // [rsp+28h] [rbp-108h]
  unsigned __int16 v44; // [rsp+28h] [rbp-108h]
  __m128i v45; // [rsp+30h] [rbp-100h] BYREF
  __int64 v46; // [rsp+40h] [rbp-F0h]
  _QWORD *v47; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v48; // [rsp+58h] [rbp-D8h]
  _QWORD v49[8]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v50; // [rsp+A0h] [rbp-90h] BYREF
  __int64 *v51; // [rsp+A8h] [rbp-88h]
  __int64 v52; // [rsp+B0h] [rbp-80h]
  int v53; // [rsp+B8h] [rbp-78h]
  char v54; // [rsp+BCh] [rbp-74h]
  __int64 v55; // [rsp+C0h] [rbp-70h] BYREF
  char v56; // [rsp+C8h] [rbp-68h] BYREF

  v6 = (_QWORD *)a1[6];
  v7 = &v55;
  v8 = a1[41];
  v9 = v6[84];
  v47 = v49;
  v50 = 0;
  v51 = &v55;
  v52 = 8;
  v53 = 0;
  v54 = 1;
  v48 = 0x800000000LL;
  if ( v9 )
  {
    if ( v8 == v9 )
    {
      v10 = 0;
    }
    else
    {
      v49[0] = v8;
      LODWORD(v48) = 1;
      HIDWORD(v52) = 1;
      v55 = v8;
      v50 = 1;
      v10 = 1;
      v7 = (__int64 *)&v56;
    }
  }
  else
  {
    v9 = v8;
    v10 = 0;
  }
  HIDWORD(v52) = v10 + 1;
  *v7 = v9;
  ++v50;
  v11 = v6[85];
  v12 = (unsigned int)v48;
  if ( v11 )
  {
    if ( (unsigned __int64)(unsigned int)v48 + 1 > HIDWORD(v48) )
    {
      sub_C8D5F0((__int64)&v47, v49, (unsigned int)v48 + 1LL, 8u, a5, a6);
      v12 = (unsigned int)v48;
    }
    v47[v12] = v11;
    LODWORD(v12) = v48 + 1;
    LODWORD(v48) = v48 + 1;
  }
  v43 = v11;
  v13 = v47;
LABEL_9:
  v14 = &v13[(unsigned int)v12];
  while ( (_DWORD)v12 )
  {
    v15 = *(v14 - 1);
    LODWORD(v12) = v12 - 1;
    --v14;
    LODWORD(v48) = v12;
    if ( v15 != v9 || v43 == v9 )
    {
      v16 = *(_QWORD *)(v15 + 112);
      v17 = *(unsigned int *)(v15 + 120);
      v18 = (__int64 *)(v16 + 8 * v17);
      v19 = (__int64 *)v16;
      if ( (__int64 *)v16 == v18 )
        goto LABEL_9;
      while ( 1 )
      {
        v20 = *v19;
        if ( !v54 )
          goto LABEL_21;
        v21 = v51;
        v15 = HIDWORD(v52);
        v17 = (__int64)&v51[HIDWORD(v52)];
        if ( v51 != (__int64 *)v17 )
        {
          while ( v20 != *v21 )
          {
            if ( (__int64 *)v17 == ++v21 )
              goto LABEL_26;
          }
LABEL_19:
          if ( v18 == ++v19 )
            goto LABEL_20;
          continue;
        }
LABEL_26:
        if ( HIDWORD(v52) < (unsigned int)v52 )
        {
          ++HIDWORD(v52);
          *(_QWORD *)v17 = v20;
          ++v50;
        }
        else
        {
LABEL_21:
          sub_C8CC70((__int64)&v50, *v19, v17, v15, v16, a6);
          if ( !(_BYTE)v17 )
            goto LABEL_19;
        }
        v22 = (unsigned int)v48;
        v15 = HIDWORD(v48);
        v23 = (unsigned int)v48 + 1LL;
        if ( v23 > HIDWORD(v48) )
        {
          sub_C8D5F0((__int64)&v47, v49, v23, 8u, v16, a6);
          v22 = (unsigned int)v48;
        }
        v17 = (__int64)v47;
        ++v19;
        v47[v22] = v20;
        LODWORD(v48) = v48 + 1;
        if ( v18 == v19 )
        {
LABEL_20:
          v13 = v47;
          LODWORD(v12) = v48;
          goto LABEL_9;
        }
      }
    }
  }
  v40 = v6[13];
  if ( v40 == v6[12] )
    goto LABEL_38;
  v24 = a1[4];
  v25 = v6[12];
  do
  {
    v26 = v51;
    if ( v54 )
      v27 = &v51[HIDWORD(v52)];
    else
      v27 = &v51[(unsigned int)v52];
    if ( v51 == v27 )
      goto LABEL_35;
    while ( 1 )
    {
      v28 = v26;
      if ( (unsigned __int64)*v26 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v27 == ++v26 )
        goto LABEL_35;
    }
    if ( v27 == v26 )
    {
LABEL_35:
      if ( !*(_BYTE *)(v25 + 9) )
        goto LABEL_36;
    }
    else
    {
      v29 = v25;
      v30 = v27;
      v31 = *v26;
      do
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v24 + 384) + 8LL * (*(_DWORD *)v29 >> 6)) & (1LL << *(_DWORD *)v29)) == 0 )
        {
          v42 = v29;
          v44 = *(_DWORD *)v29;
          v37 = sub_2E31DD0(v31, *(_DWORD *)v29, -1, -1);
          v29 = v42;
          if ( !v37 )
          {
            v45.m128i_i32[0] = v44;
            v45.m128i_i64[1] = -1;
            v46 = -1;
            sub_35AE1F0((unsigned __int64 *)(v31 + 184), &v45);
            v29 = v42;
          }
        }
        v32 = v28 + 1;
        if ( v28 + 1 == v30 )
          break;
        while ( 1 )
        {
          v31 = *v32;
          v28 = v32;
          if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v30 == ++v32 )
            goto LABEL_48;
        }
      }
      while ( v30 != v32 );
LABEL_48:
      v25 = v29;
      if ( !*(_BYTE *)(v29 + 9) )
        goto LABEL_36;
    }
    v33 = a1 + 40;
    if ( a1 + 40 == (_QWORD *)a1[41] )
      goto LABEL_36;
    v41 = v24;
    v34 = a1[41];
    do
    {
      while ( !v54 )
      {
        if ( !sub_C8CA60((__int64)&v50, v34) )
          goto LABEL_64;
LABEL_56:
        v34 = *(_QWORD *)(v34 + 8);
        if ( v33 == (_QWORD *)v34 )
          goto LABEL_57;
      }
      v35 = v51;
      v36 = &v51[HIDWORD(v52)];
      if ( v51 != v36 )
      {
        while ( v34 != *v35 )
        {
          if ( v36 == ++v35 )
            goto LABEL_64;
        }
        goto LABEL_56;
      }
LABEL_64:
      v38 = *(_DWORD *)(v25 + 4);
      if ( (unsigned __int8)sub_2E31DD0(v34, v38, -1, -1) )
        goto LABEL_56;
      v45.m128i_i64[1] = -1;
      v45.m128i_i32[0] = (unsigned __int16)v38;
      v46 = -1;
      sub_35AE1F0((unsigned __int64 *)(v34 + 184), &v45);
      v34 = *(_QWORD *)(v34 + 8);
    }
    while ( v33 != (_QWORD *)v34 );
LABEL_57:
    v24 = v41;
LABEL_36:
    v25 += 12;
  }
  while ( v40 != v25 );
  v13 = v47;
LABEL_38:
  if ( v13 != v49 )
    _libc_free((unsigned __int64)v13);
  if ( !v54 )
    _libc_free((unsigned __int64)v51);
}
