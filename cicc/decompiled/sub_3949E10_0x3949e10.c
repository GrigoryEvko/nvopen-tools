// Function: sub_3949E10
// Address: 0x3949e10
//
__int64 __fastcall sub_3949E10(unsigned __int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  char v5; // al
  unsigned __int64 v6; // r8
  __int64 v7; // r12
  __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 *v11; // rdi
  __int64 v12; // rax
  size_t v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rsi
  size_t v16; // rdx
  char v18; // [rsp+17h] [rbp-129h]
  char v19; // [rsp+17h] [rbp-129h]
  __int64 v20; // [rsp+18h] [rbp-128h]
  char v21; // [rsp+28h] [rbp-118h]
  unsigned __int8 v22; // [rsp+28h] [rbp-118h]
  __int64 v23[2]; // [rsp+30h] [rbp-110h] BYREF
  char v24; // [rsp+40h] [rbp-100h]
  const char *v25; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v26; // [rsp+58h] [rbp-E8h]
  __int16 v27; // [rsp+60h] [rbp-E0h]
  const char **v28; // [rsp+70h] [rbp-D0h] BYREF
  const char *v29; // [rsp+78h] [rbp-C8h]
  __int16 v30; // [rsp+80h] [rbp-C0h]
  const char ***v31; // [rsp+90h] [rbp-B0h] BYREF
  __m128i **v32; // [rsp+98h] [rbp-A8h]
  __int16 v33; // [rsp+A0h] [rbp-A0h]
  unsigned __int64 v34; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v35; // [rsp+B8h] [rbp-88h]
  __int64 v36; // [rsp+C0h] [rbp-80h]
  __m128i *v37[2]; // [rsp+D0h] [rbp-70h] BYREF
  _BYTE v38[16]; // [rsp+E0h] [rbp-60h] BYREF
  __int64 *v39; // [rsp+F0h] [rbp-50h] BYREF
  size_t n; // [rsp+F8h] [rbp-48h]
  _QWORD src[8]; // [rsp+100h] [rbp-40h] BYREF

  v36 = 0x1000000000LL;
  v3 = *a2;
  v4 = a2[1];
  v34 = 0;
  v35 = 0;
  v20 = v4;
  if ( v3 == v4 )
  {
    v6 = 0;
    v5 = 1;
    goto LABEL_19;
  }
  while ( 1 )
  {
    LOWORD(src[0]) = 260;
    v39 = (__int64 *)v3;
    sub_16C2DE0((__int64)v23, (__int64)&v39, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
    if ( (v24 & 1) != 0 && LODWORD(v23[0]) )
    {
      (*(void (__fastcall **)(__m128i **))(*(_QWORD *)v23[1] + 32LL))(v37);
      v25 = "can't open file '";
      v28 = &v25;
      v29 = "': ";
      v26 = v3;
      v31 = &v28;
      v27 = 1027;
      v30 = 770;
      v32 = v37;
      v33 = 1026;
      sub_16E2FC0((__int64 *)&v39, (__int64)&v31);
      v11 = (__int64 *)*a3;
      v12 = (__int64)v39;
      if ( v39 != src )
      {
        v13 = n;
        v14 = src[0];
        if ( v11 != a3 + 2 )
        {
          v15 = a3[2];
          *a3 = (__int64)v39;
          a3[1] = v13;
          a3[2] = v14;
          if ( !v11 )
            goto LABEL_37;
LABEL_23:
          v39 = v11;
          src[0] = v15;
          goto LABEL_24;
        }
LABEL_36:
        *a3 = v12;
        a3[1] = v13;
        a3[2] = v14;
        goto LABEL_37;
      }
LABEL_38:
      v16 = n;
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)v11 = src[0];
        else
          memcpy(v11, src, n);
        v16 = n;
        v11 = (__int64 *)*a3;
      }
      a3[1] = v16;
      *((_BYTE *)v11 + v16) = 0;
      v11 = v39;
      goto LABEL_24;
    }
    v37[0] = (__m128i *)v38;
    v37[1] = 0;
    v38[0] = 0;
    v5 = sub_39483B0(a1, v23[0], (__int64)&v34, v37);
    if ( !v5 )
      break;
    if ( v37[0] != (__m128i *)v38 )
    {
      v18 = v5;
      j_j___libc_free_0((unsigned __int64)v37[0]);
      v5 = v18;
    }
    if ( (v24 & 1) != 0 || !v23[0] )
    {
      v3 += 32;
      if ( v20 == v3 )
        goto LABEL_12;
    }
    else
    {
      v19 = v5;
      v3 += 32;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v23[0] + 8LL))(v23[0]);
      v5 = v19;
      if ( v20 == v3 )
        goto LABEL_12;
    }
  }
  v25 = "error parsing file '";
  v28 = &v25;
  v29 = "': ";
  v30 = 770;
  v33 = 1026;
  v26 = v3;
  v31 = &v28;
  v27 = 1027;
  v32 = v37;
  sub_16E2FC0((__int64 *)&v39, (__int64)&v31);
  v11 = (__int64 *)*a3;
  v12 = (__int64)v39;
  if ( v39 == src )
    goto LABEL_38;
  v13 = n;
  v14 = src[0];
  if ( v11 == a3 + 2 )
    goto LABEL_36;
  v15 = a3[2];
  *a3 = (__int64)v39;
  a3[1] = v13;
  a3[2] = v14;
  if ( v11 )
    goto LABEL_23;
LABEL_37:
  v39 = src;
  v11 = src;
LABEL_24:
  n = 0;
  *(_BYTE *)v11 = 0;
  if ( v39 != src )
    j_j___libc_free_0((unsigned __int64)v39);
  if ( v37[0] != (__m128i *)v38 )
    j_j___libc_free_0((unsigned __int64)v37[0]);
  if ( (v24 & 1) != 0 || !v23[0] )
  {
    v5 = 0;
  }
  else
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v23[0] + 8LL))(v23[0]);
    v5 = 0;
  }
LABEL_12:
  v6 = v34;
  if ( HIDWORD(v35) && (_DWORD)v35 )
  {
    v7 = 8LL * (unsigned int)v35;
    v8 = 0;
    do
    {
      v9 = *(_QWORD *)(v6 + v8);
      if ( v9 != -8 && v9 )
      {
        v21 = v5;
        _libc_free(v9);
        v6 = v34;
        v5 = v21;
      }
      v8 += 8;
    }
    while ( v7 != v8 );
  }
LABEL_19:
  v22 = v5;
  _libc_free(v6);
  return v22;
}
