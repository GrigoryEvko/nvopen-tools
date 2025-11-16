// Function: sub_2B1B4A0
// Address: 0x2b1b4a0
//
void __fastcall sub_2B1B4A0(__int64 a1, __int64 a2, __int64 a3, const void *a4, unsigned __int64 a5, __int64 a6)
{
  size_t v6; // r15
  _QWORD *v11; // rdi
  __int64 v12; // r8
  int v13; // r13d
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  void **p_s; // r10
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rdi
  unsigned __int64 *v22; // rdx
  unsigned __int64 *v23; // r12
  int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-98h]
  __int64 v26; // [rsp+10h] [rbp-90h]
  __int64 v27; // [rsp+18h] [rbp-88h]
  __int64 v28; // [rsp+18h] [rbp-88h]
  void **v29; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v30; // [rsp+18h] [rbp-88h]
  int v31; // [rsp+18h] [rbp-88h]
  unsigned __int64 v32; // [rsp+28h] [rbp-78h] BYREF
  void *s; // [rsp+30h] [rbp-70h] BYREF
  __int64 v34; // [rsp+38h] [rbp-68h]
  _BYTE v35[96]; // [rsp+40h] [rbp-60h] BYREF

  v6 = 8 * a5;
  v11 = (_QWORD *)(a1 + 32);
  v12 = (__int64)(8 * a5) >> 3;
  *(v11 - 4) = a2;
  *(v11 - 3) = a3;
  *(_QWORD *)(a1 + 16) = v11;
  *(_QWORD *)(a1 + 24) = 0x600000000LL;
  if ( v6 > 0x30 )
  {
    v27 = v12;
    sub_C8D5F0(a1 + 16, v11, v12, 8u, v12, a1 + 16);
    v12 = v27;
    v11 = (_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * *(unsigned int *)(a1 + 24));
LABEL_12:
    v28 = v12;
    memcpy(v11, a4, v6);
    v12 = v28;
    s = v35;
    *(_DWORD *)(a1 + 24) += v28;
    v34 = 0x600000000LL;
    if ( a5 > 6 )
    {
      sub_C8D5F0((__int64)&s, v35, a5, 8u, v28, a6);
      memset(s, 0, v6);
      goto LABEL_5;
    }
    goto LABEL_4;
  }
  if ( v6 )
    goto LABEL_12;
  *(_DWORD *)(a1 + 24) = v12;
  s = v35;
  v34 = 0x600000000LL;
  if ( a5 > 6 )
  {
    sub_C8D5F0((__int64)&s, v35, a5, 8u, v12, a6);
    goto LABEL_5;
  }
LABEL_4:
  if ( a5 && v6 )
    memset(v35, 0, v6);
LABEL_5:
  v13 = *(_DWORD *)(a3 + 4);
  LODWORD(v34) = a5;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  v14 = v13 & 0x7FFFFFF;
  *(_QWORD *)(a1 + 88) = 0x100000000LL;
  if ( v14 > 1uLL )
  {
    v15 = sub_C8D7D0(a1 + 80, a1 + 96, v14, 0x40u, &v32, a6);
    p_s = &s;
    v19 = v14;
    v25 = v15;
    v20 = v15;
    do
    {
      while ( 1 )
      {
        if ( v20 )
        {
          *(_DWORD *)(v20 + 8) = 0;
          *(_QWORD *)v20 = v20 + 16;
          *(_DWORD *)(v20 + 12) = 6;
          if ( (_DWORD)v34 )
            break;
        }
        v20 += 64;
        if ( !--v19 )
          goto LABEL_23;
      }
      v21 = v20;
      v26 = v19;
      v20 += 64;
      v29 = p_s;
      sub_2B0CFB0(v21, (__int64)p_s, v19, (unsigned int)v34, v16, v17);
      p_s = v29;
      v19 = v26 - 1;
    }
    while ( v26 != 1 );
LABEL_23:
    v22 = *(unsigned __int64 **)(a1 + 80);
    v23 = &v22[8 * (unsigned __int64)*(unsigned int *)(a1 + 88)];
    if ( v22 != v23 )
    {
      do
      {
        v23 -= 8;
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
        {
          v30 = v22;
          _libc_free(*v23);
          v22 = v30;
        }
      }
      while ( v22 != v23 );
      v23 = *(unsigned __int64 **)(a1 + 80);
    }
    v24 = v32;
    if ( (unsigned __int64 *)(a1 + 96) != v23 )
    {
      v31 = v32;
      _libc_free((unsigned __int64)v23);
      v24 = v31;
    }
    *(_DWORD *)(a1 + 92) = v24;
    *(_DWORD *)(a1 + 88) = v14;
    *(_QWORD *)(a1 + 80) = v25;
  }
  else
  {
    if ( v14 )
    {
      *(_QWORD *)(a1 + 96) = a1 + 112;
      *(_QWORD *)(a1 + 104) = 0x600000000LL;
      if ( (_DWORD)v34 )
        sub_2B0CFB0(a1 + 96, (__int64)&s, v14, (__int64)a4, v12, a6);
    }
    *(_DWORD *)(a1 + 88) = v14;
  }
  if ( s != v35 )
    _libc_free((unsigned __int64)s);
}
