// Function: sub_BD79D0
// Address: 0xbd79d0
//
__int64 __fastcall sub_BD79D0(
        __int64 *a1,
        __int64 *a2,
        unsigned __int8 (__fastcall *a3)(__int64, __int64 *),
        __int64 a4)
{
  __int64 *v6; // rbx
  __int64 *v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int i; // eax
  __int64 *v12; // rcx
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 result; // rax
  __int64 *v19; // r12
  __int64 **v20; // rax
  __int64 **v21; // rdx
  __int64 v22; // rdx
  unsigned __int64 *v23; // r10
  int v24; // eax
  unsigned __int64 *v25; // rdi
  unsigned __int64 v26; // rax
  char v27; // dl
  __int64 *v28; // [rsp+0h] [rbp-1A0h]
  char *v29; // [rsp+0h] [rbp-1A0h]
  __int64 *v30; // [rsp+10h] [rbp-190h]
  unsigned __int64 v32; // [rsp+20h] [rbp-180h] BYREF
  __int64 v33; // [rsp+28h] [rbp-178h]
  __int64 *v34; // [rsp+30h] [rbp-170h]
  __int64 v35; // [rsp+40h] [rbp-160h] BYREF
  __int64 **v36; // [rsp+48h] [rbp-158h]
  __int64 v37; // [rsp+50h] [rbp-150h]
  int v38; // [rsp+58h] [rbp-148h]
  char v39; // [rsp+5Ch] [rbp-144h]
  char v40; // [rsp+60h] [rbp-140h] BYREF
  __int64 *v41; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-F8h]
  _BYTE v43[240]; // [rsp+B0h] [rbp-F0h] BYREF

  v6 = a2;
  v7 = (__int64 *)a1[2];
  v41 = (__int64 *)v43;
  v42 = 0x800000000LL;
  v36 = (__int64 **)&v40;
  v35 = 0;
  v37 = 8;
  v38 = 0;
  v39 = 1;
  v30 = a2 + 2;
  if ( !v7 )
    goto LABEL_25;
  do
  {
    v8 = v7;
    v7 = (__int64 *)v7[1];
    a2 = v8;
    if ( !a3(a4, v8) )
      continue;
    a2 = (__int64 *)v8[3];
    if ( (unsigned __int8)(*(_BYTE *)a2 - 4) > 0x11u )
    {
      if ( *v8 )
      {
        a2 = (__int64 *)v8[2];
        v9 = v8[1];
        *a2 = v9;
        if ( v9 )
        {
          a2 = (__int64 *)v8[2];
          *(_QWORD *)(v9 + 16) = a2;
        }
      }
      *v8 = (__int64)v6;
      if ( v6 )
      {
        v10 = v6[2];
        v8[1] = v10;
        if ( v10 )
        {
          a2 = v8 + 1;
          *(_QWORD *)(v10 + 16) = v8 + 1;
        }
        v8[2] = (__int64)v30;
        v6[2] = (__int64)v8;
      }
      continue;
    }
    if ( !v39 )
      goto LABEL_52;
    v20 = v36;
    v21 = &v36[HIDWORD(v37)];
    if ( v36 == v21 )
    {
LABEL_38:
      if ( HIDWORD(v37) >= (unsigned int)v37 )
      {
LABEL_52:
        v28 = (__int64 *)v8[3];
        sub_C8CC70(&v35, a2);
        a2 = v28;
        if ( !v27 )
          continue;
      }
      else
      {
        ++HIDWORD(v37);
        *v21 = a2;
        ++v35;
      }
      v34 = a2;
      v32 = 6;
      v33 = 0;
      if ( a2 != (__int64 *)-8192LL && a2 != (__int64 *)-4096LL )
        sub_BD73F0((__int64)&v32);
      v22 = (unsigned int)v42;
      v23 = &v32;
      a2 = v41;
      v24 = v42;
      if ( (unsigned __int64)(unsigned int)v42 + 1 > HIDWORD(v42) )
      {
        if ( v41 > (__int64 *)&v32 || &v32 >= (unsigned __int64 *)&v41[3 * (unsigned int)v42] )
        {
          sub_BD61F0((__int64)&v41, (unsigned int)v42 + 1LL);
          v22 = (unsigned int)v42;
          a2 = v41;
          v23 = &v32;
          v24 = v42;
        }
        else
        {
          v29 = (char *)((char *)&v32 - (char *)v41);
          sub_BD61F0((__int64)&v41, (unsigned int)v42 + 1LL);
          a2 = v41;
          v22 = (unsigned int)v42;
          v23 = (unsigned __int64 *)&v29[(_QWORD)v41];
          v24 = v42;
        }
      }
      v25 = (unsigned __int64 *)&a2[3 * v22];
      if ( v25 )
      {
        *v25 = 6;
        v26 = v23[2];
        v25[1] = 0;
        v25[2] = v26;
        if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
        {
          a2 = (__int64 *)(*v23 & 0xFFFFFFFFFFFFFFF8LL);
          sub_BD6050(v25, (unsigned __int64)a2);
        }
        v24 = v42;
      }
      LODWORD(v42) = v24 + 1;
      LOBYTE(a2) = v34 + 512 != 0;
      if ( ((v34 != 0) & (unsigned __int8)a2) != 0 && v34 != (__int64 *)-8192LL )
        sub_BD60C0(&v32);
      continue;
    }
    while ( a2 != *v20 )
    {
      if ( v21 == ++v20 )
        goto LABEL_38;
    }
  }
  while ( v7 );
  for ( i = v42; (_DWORD)v42; i = v42 )
  {
    v12 = v41;
    v32 = 6;
    v33 = 0;
    v13 = &v41[3 * i - 3];
    v34 = (__int64 *)v13[2];
    if ( v34 != 0 && v34 + 512 != 0 && v34 != (__int64 *)-8192LL )
    {
      sub_BD6050(&v32, *v13 & 0xFFFFFFFFFFFFFFF8LL);
      v12 = v41;
      i = v42;
    }
    v14 = i - 1;
    LODWORD(v42) = v14;
    v15 = &v12[3 * v14];
    v16 = v15[2];
    if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
      sub_BD60C0(v15);
    a2 = a1;
    sub_ADBE50(v34, a1, v6);
    if ( v34 + 512 != 0 && v34 != 0 && v34 != (__int64 *)-8192LL )
      sub_BD60C0(&v32);
  }
  if ( !v39 )
    _libc_free(v36, a2);
LABEL_25:
  v17 = v41;
  result = 3LL * (unsigned int)v42;
  v19 = &v41[3 * (unsigned int)v42];
  if ( v41 != v19 )
  {
    do
    {
      result = *(v19 - 1);
      v19 -= 3;
      if ( result != -4096 && result != 0 && result != -8192 )
        result = sub_BD60C0(v19);
    }
    while ( v17 != v19 );
    v19 = v41;
  }
  if ( v19 != (__int64 *)v43 )
    return _libc_free(v19, a2);
  return result;
}
