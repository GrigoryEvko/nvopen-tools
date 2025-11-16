// Function: sub_2DF2100
// Address: 0x2df2100
//
_QWORD *__fastcall sub_2DF2100(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  _QWORD *v3; // r13
  const char **v4; // rax
  size_t v5; // rdx
  const char *v6; // r14
  __int64 *v7; // rsi
  _BYTE *v8; // rbx
  unsigned __int8 v9; // al
  _BYTE *v10; // r14
  _BYTE *v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  _BYTE **v15; // r14
  const char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r14
  __int64 v26; // r15
  unsigned __int64 v27; // r12
  __int64 v28; // rsi
  char *v29; // rbx
  char *v30; // r12
  __int64 v31; // rsi
  unsigned __int64 v32; // rbx
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rdi
  char *v35; // rbx
  char *v36; // r12
  __int64 v37; // rsi
  char *v38; // rbx
  char *v39; // r12
  __int64 v40; // rsi
  char *v41; // rbx
  char *v42; // r12
  __int64 v43; // rsi
  __int64 v45; // [rsp+8h] [rbp-1F8h]
  __int64 v46; // [rsp+18h] [rbp-1E8h]
  _QWORD v47[4]; // [rsp+20h] [rbp-1E0h] BYREF
  __int16 v48; // [rsp+40h] [rbp-1C0h]
  char *v49; // [rsp+58h] [rbp-1A8h]
  int v50; // [rsp+60h] [rbp-1A0h]
  char v51; // [rsp+68h] [rbp-198h] BYREF
  char *v52; // [rsp+88h] [rbp-178h]
  int v53; // [rsp+90h] [rbp-170h]
  char v54; // [rsp+98h] [rbp-168h] BYREF
  char *v55; // [rsp+B8h] [rbp-148h]
  char v56; // [rsp+C8h] [rbp-138h] BYREF
  char *v57; // [rsp+E8h] [rbp-118h]
  char v58; // [rsp+F8h] [rbp-108h] BYREF
  char *v59; // [rsp+118h] [rbp-E8h]
  int v60; // [rsp+120h] [rbp-E0h]
  char v61; // [rsp+128h] [rbp-D8h] BYREF
  __int64 v62; // [rsp+150h] [rbp-B0h]
  unsigned int v63; // [rsp+160h] [rbp-A0h]
  unsigned __int64 v64; // [rsp+168h] [rbp-98h]
  unsigned int v65; // [rsp+170h] [rbp-90h]
  char *v66; // [rsp+178h] [rbp-88h] BYREF
  int v67; // [rsp+180h] [rbp-80h]
  char v68; // [rsp+188h] [rbp-78h] BYREF
  __int64 v69; // [rsp+1B8h] [rbp-48h]
  unsigned int v70; // [rsp+1C8h] [rbp-38h]

  BYTE4(v46) = 0;
  v1 = sub_ACD640(**(_QWORD **)(a1 + 8), 1, 0);
  v2 = *(_QWORD *)(a1 + 16);
  v48 = 260;
  v47[0] = v2;
  v3 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v3 )
    sub_B30000((__int64)v3, *(_QWORD *)a1, **(_QWORD ***)(a1 + 8), 0, 7, v1, (__int64)v47, 0, 0, v46, 0);
  v4 = *(const char ***)(a1 + 24);
  v5 = 0;
  v6 = *v4;
  if ( *v4 )
    v5 = strlen(*v4);
  sub_B31A00((__int64)v3, (__int64)v6, v5);
  sub_B2F770((__int64)v3, 0);
  v7 = (__int64 *)v3[5];
  *((_BYTE *)v3 + 32) = v3[4] & 0x3F | 0x80;
  v8 = **(_BYTE ***)(a1 + 32);
  v9 = *(v8 - 16);
  v10 = v8 - 16;
  if ( (v9 & 2) != 0 )
    v11 = (_BYTE *)*((_QWORD *)v8 - 4);
  else
    v11 = &v10[-8 * ((v9 >> 2) & 0xF)];
  v12 = *((_QWORD *)v11 + 5);
  sub_AE0470((__int64)v47, v7, 0, v12);
  v13 = sub_ADC9A0((__int64)v47, (__int64)"unsigned char", 13, 8, 8, 64, 0);
  if ( *v8 != 16 )
  {
    v14 = *(v8 - 16);
    if ( (v14 & 2) != 0 )
      v15 = (_BYTE **)*((_QWORD *)v8 - 4);
    else
      v15 = (_BYTE **)&v10[-8 * ((v14 >> 2) & 0xF)];
    v8 = *v15;
  }
  v45 = v13;
  v16 = sub_BD5D20((__int64)v3);
  v18 = sub_ADD600((__int64)v47, v12, (__int64)v16, v17, 0, 0, (__int64)v8, 0, v45, 1u, 1u, 0, 0, 0, 0, 0);
  sub_B994D0((__int64)v3, 0, v18);
  sub_ADCDB0((__int64)v47, 0, v19, v20, v21, v22);
  v23 = v70;
  if ( v70 )
  {
    v24 = v69;
    v25 = v69 + 56LL * v70;
    do
    {
      if ( *(_QWORD *)v24 != -4096 && *(_QWORD *)v24 != -8192 )
      {
        v26 = *(_QWORD *)(v24 + 8);
        v27 = v26 + 8LL * *(unsigned int *)(v24 + 16);
        if ( v26 != v27 )
        {
          do
          {
            v28 = *(_QWORD *)(v27 - 8);
            v27 -= 8LL;
            if ( v28 )
              sub_B91220(v27, v28);
          }
          while ( v26 != v27 );
          v27 = *(_QWORD *)(v24 + 8);
        }
        if ( v27 != v24 + 24 )
          _libc_free(v27);
      }
      v24 += 56;
    }
    while ( v25 != v24 );
    v23 = v70;
  }
  sub_C7D6A0(v69, 56 * v23, 8);
  v29 = v66;
  v30 = &v66[8 * v67];
  if ( v66 != v30 )
  {
    do
    {
      v31 = *((_QWORD *)v30 - 1);
      v30 -= 8;
      if ( v31 )
        sub_B91220((__int64)v30, v31);
    }
    while ( v29 != v30 );
    v30 = v66;
  }
  if ( v30 != &v68 )
    _libc_free((unsigned __int64)v30);
  v32 = v64;
  v33 = v64 + 56LL * v65;
  if ( v64 != v33 )
  {
    do
    {
      v33 -= 56LL;
      v34 = *(_QWORD *)(v33 + 40);
      if ( v34 != v33 + 56 )
        _libc_free(v34);
      sub_C7D6A0(*(_QWORD *)(v33 + 16), 8LL * *(unsigned int *)(v33 + 32), 8);
    }
    while ( v32 != v33 );
    v33 = v64;
  }
  if ( (char **)v33 != &v66 )
    _libc_free(v33);
  sub_C7D6A0(v62, 16LL * v63, 8);
  v35 = v59;
  v36 = &v59[8 * v60];
  if ( v59 != v36 )
  {
    do
    {
      v37 = *((_QWORD *)v36 - 1);
      v36 -= 8;
      if ( v37 )
        sub_B91220((__int64)v36, v37);
    }
    while ( v35 != v36 );
    v36 = v59;
  }
  if ( v36 != &v61 )
    _libc_free((unsigned __int64)v36);
  if ( v57 != &v58 )
    _libc_free((unsigned __int64)v57);
  if ( v55 != &v56 )
    _libc_free((unsigned __int64)v55);
  v38 = v52;
  v39 = &v52[8 * v53];
  if ( v52 != v39 )
  {
    do
    {
      v40 = *((_QWORD *)v39 - 1);
      v39 -= 8;
      if ( v40 )
        sub_B91220((__int64)v39, v40);
    }
    while ( v38 != v39 );
    v39 = v52;
  }
  if ( v39 != &v54 )
    _libc_free((unsigned __int64)v39);
  v41 = v49;
  v42 = &v49[8 * v50];
  if ( v49 != v42 )
  {
    do
    {
      v43 = *((_QWORD *)v42 - 1);
      v42 -= 8;
      if ( v43 )
        sub_B91220((__int64)v42, v43);
    }
    while ( v41 != v42 );
    v42 = v49;
  }
  if ( v42 != &v51 )
    _libc_free((unsigned __int64)v42);
  return v3;
}
