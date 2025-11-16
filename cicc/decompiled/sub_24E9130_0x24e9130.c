// Function: sub_24E9130
// Address: 0x24e9130
//
void __fastcall sub_24E9130(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  const __m128i *v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r12
  unsigned __int64 v11; // rax
  int v12; // edx
  _QWORD *v13; // rdi
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r12
  __int64 v20; // rbx
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  _QWORD *v28; // rax
  __int64 v29; // r12
  __int64 v30; // r13
  __int64 v31; // r8
  __int64 v32; // r13
  __int64 v33; // rdx
  _BYTE *v34; // r15
  _BYTE *v35; // r13
  unsigned __int64 v36; // r14
  unsigned __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rbx
  __int64 v40; // r14
  __int64 i; // r15
  __int64 v42; // rax
  _QWORD *v43; // r10
  __int64 v44; // rax
  __int64 v45; // rcx
  unsigned int v46; // eax
  __int16 v47; // dx
  unsigned __int64 *v48; // r11
  char v49; // al
  char v50; // dl
  __int64 v51; // [rsp+8h] [rbp-118h]
  _QWORD *v52; // [rsp+10h] [rbp-110h]
  _QWORD *v53; // [rsp+20h] [rbp-100h]
  __int64 v54; // [rsp+28h] [rbp-F8h]
  _QWORD *v55; // [rsp+28h] [rbp-F8h]
  __int64 v56; // [rsp+30h] [rbp-F0h]
  unsigned int **v57; // [rsp+38h] [rbp-E8h]
  __m128i v58[2]; // [rsp+40h] [rbp-E0h] BYREF
  char v59; // [rsp+60h] [rbp-C0h]
  char v60; // [rsp+61h] [rbp-BFh]
  __m128i v61; // [rsp+70h] [rbp-B0h] BYREF
  char v62; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE *v63; // [rsp+88h] [rbp-98h]
  __int64 v64; // [rsp+90h] [rbp-90h]
  _BYTE v65[56]; // [rsp+98h] [rbp-88h] BYREF
  __int64 v66; // [rsp+D0h] [rbp-50h]
  __int64 v67; // [rsp+D8h] [rbp-48h]
  char v68; // [rsp+E0h] [rbp-40h]
  __int64 v69; // [rsp+E4h] [rbp-3Ch]

  v1 = a1;
  v56 = (__int64)(a1 + 25);
  v61.m128i_i64[0] = *(_QWORD *)(a1[3] + 320LL);
  v2 = sub_24E84F0((__int64)(a1 + 25), v61.m128i_i64);
  v5 = (const __m128i *)a1[2];
  v6 = v2[2];
  v7 = a1[35];
  v60 = 1;
  v59 = 3;
  v8 = *(_QWORD *)(v7 + 80);
  v9 = v8 - 24;
  if ( !v8 )
    v9 = 0;
  v54 = v9;
  v10 = v9;
  v58[0].m128i_i64[0] = (__int64)"entry";
  sub_9C6370(&v61, v58, v5, v8, v3, v4);
  sub_BD6B50((unsigned __int8 *)v6, (const char **)&v61);
  sub_AA4AC0(v6, v10 + 24);
  v11 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 + 48 == v11 )
  {
    v13 = 0;
  }
  else
  {
    if ( !v11 )
      BUG();
    v12 = *(unsigned __int8 *)(v11 - 24);
    v13 = 0;
    v14 = (_QWORD *)(v11 - 24);
    if ( (unsigned int)(v12 - 30) < 0xB )
      v13 = v14;
  }
  sub_B43D60(v13);
  v57 = (unsigned int **)(v1 + 5);
  v53 = *(_QWORD **)(*(_QWORD *)(v6 + 16) + 24LL);
  sub_D5F1F0((__int64)(v1 + 5), (__int64)v53);
  LOWORD(v64) = 257;
  v15 = sub_BD2C40(72, unk_3F148B8);
  v16 = (__int64)v15;
  if ( v15 )
    sub_B4C8A0((__int64)v15, v1[14], 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(*(_QWORD *)v1[16] + 16LL))(
    v1[16],
    v16,
    &v61,
    v1[12],
    v1[13]);
  v17 = v1[5];
  v18 = 16LL * *((unsigned int *)v1 + 12);
  if ( v17 != v17 + v18 )
  {
    v52 = v1;
    v19 = v17 + v18;
    v20 = v1[5];
    do
    {
      v21 = *(_QWORD *)(v20 + 8);
      v22 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0(v16, v22, v21);
    }
    while ( v19 != v20 );
    v1 = v52;
  }
  sub_B43D60(v53);
  v1[11] = v6;
  *((_WORD *)v1 + 52) = 0;
  v23 = v1[3];
  v1[12] = v6 + 48;
  v24 = *(_DWORD *)(v23 + 280);
  if ( v24 )
  {
    if ( (unsigned int)(v24 - 1) > 2 )
      goto LABEL_21;
    v61.m128i_i64[0] = v1[37];
    v25 = sub_24E84F0(v56, v61.m128i_i64)[2];
    v26 = *(_QWORD *)(v25 + 32);
    if ( v26 == *(_QWORD *)(v25 + 40) + 48LL || !v26 )
      BUG();
    v27 = *(_QWORD *)(v26 - 56);
  }
  else
  {
    v61.m128i_i64[0] = *(_QWORD *)(v23 + 344);
    v27 = sub_24E84F0(v56, v61.m128i_i64)[2];
  }
  LOWORD(v64) = 257;
  v28 = sub_BD2C40(72, 1u);
  v29 = (__int64)v28;
  if ( v28 )
    sub_B4C8F0((__int64)v28, v27, 1u, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __m128i *, unsigned int *, unsigned int *))(*(_QWORD *)v1[16] + 16LL))(
    v1[16],
    v29,
    &v61,
    v57[7],
    v57[8]);
  sub_94AAF0(v57, v29);
LABEL_21:
  v30 = *(_QWORD *)(v54 + 72);
  v69 = 0;
  v61.m128i_i64[1] = 0x100000000LL;
  v67 = v30;
  v61.m128i_i64[0] = (__int64)&v62;
  v63 = v65;
  v64 = 0x600000000LL;
  v66 = 0;
  v68 = 0;
  HIDWORD(v69) = *(_DWORD *)(v30 + 92);
  sub_B1F440((__int64)&v61);
  v31 = v30 + 72;
  v32 = *(_QWORD *)(v30 + 80);
  if ( v31 != v32 )
  {
    if ( !v32 )
      BUG();
    while ( 1 )
    {
      v33 = *(_QWORD *)(v32 + 32);
      if ( v33 != v32 + 24 )
        break;
      v32 = *(_QWORD *)(v32 + 8);
      if ( v31 == v32 )
        goto LABEL_27;
      if ( !v32 )
        BUG();
    }
    if ( v31 != v32 )
    {
      v38 = v6;
      v39 = v51;
      v40 = v31;
LABEL_41:
      for ( i = *(_QWORD *)(v33 + 8); ; i = *(_QWORD *)(v32 + 32) )
      {
        v42 = v32 - 24;
        if ( !v32 )
          v42 = 0;
        if ( i != v42 + 48 )
        {
          if ( *(_BYTE *)(v33 - 24) == 60 )
          {
            v43 = (_QWORD *)(v33 - 24);
            if ( *(_QWORD *)(v33 - 8) )
            {
LABEL_50:
              v44 = *(_QWORD *)(v33 + 16);
              if ( v44 )
              {
                v45 = (unsigned int)(*(_DWORD *)(v44 + 44) + 1);
                v46 = *(_DWORD *)(v44 + 44) + 1;
              }
              else
              {
                v45 = 0;
                v46 = 0;
              }
              if ( (v46 >= (unsigned int)v64 || !*(_QWORD *)&v63[8 * v45]) && **(_BYTE **)(v33 - 56) == 17 )
              {
                v55 = v43;
                v48 = (unsigned __int64 *)sub_AA5190(v38);
                if ( v48 )
                {
                  v49 = v47;
                  v50 = HIBYTE(v47);
                }
                else
                {
                  v50 = 0;
                  v49 = 0;
                }
                LOBYTE(v39) = v49;
                BYTE1(v39) = v50;
                sub_B44550(v55, v38, v48, v39);
              }
            }
          }
          if ( v32 == v40 )
            goto LABEL_27;
          v33 = i;
          goto LABEL_41;
        }
        v32 = *(_QWORD *)(v32 + 8);
        if ( v40 == v32 )
          break;
        if ( !v32 )
          BUG();
      }
      if ( *(_BYTE *)(v33 - 24) == 60 )
      {
        v43 = (_QWORD *)(v33 - 24);
        if ( *(_QWORD *)(v33 - 8) )
          goto LABEL_50;
      }
    }
  }
LABEL_27:
  v34 = v63;
  v35 = &v63[8 * (unsigned int)v64];
  if ( v63 != v35 )
  {
    do
    {
      v36 = *((_QWORD *)v35 - 1);
      v35 -= 8;
      if ( v36 )
      {
        v37 = *(_QWORD *)(v36 + 24);
        if ( v37 != v36 + 40 )
          _libc_free(v37);
        j_j___libc_free_0(v36);
      }
    }
    while ( v34 != v35 );
    v35 = v63;
  }
  if ( v35 != v65 )
    _libc_free((unsigned __int64)v35);
  if ( (char *)v61.m128i_i64[0] != &v62 )
    _libc_free(v61.m128i_u64[0]);
}
