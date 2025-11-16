// Function: sub_14A0CC0
// Address: 0x14a0cc0
//
__int64 __fastcall sub_14A0CC0(__int64 *a1, unsigned __int8 a2, char a3)
{
  __int64 v6; // rax
  _BYTE *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rsi
  int v11; // edx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // r13
  unsigned int v16; // r14d
  size_t v18; // rdx
  int v19; // [rsp+8h] [rbp-118h]
  void *s2; // [rsp+10h] [rbp-110h] BYREF
  __int64 v21; // [rsp+18h] [rbp-108h]
  _BYTE v22[64]; // [rsp+20h] [rbp-100h] BYREF
  void *s1; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+68h] [rbp-B8h]
  _BYTE s[176]; // [rsp+70h] [rbp-B0h] BYREF

  v6 = *a1;
  s1 = s;
  v24 = 0x2000000000LL;
  v7 = s;
  v8 = *(_QWORD *)(v6 + 32);
  v9 = (unsigned int)v8;
  if ( (unsigned int)v8 > 0x20 )
  {
    v19 = v8;
    sub_16CD150(&s1, s, (unsigned int)v8, 4);
    v7 = s1;
    LODWORD(v8) = v19;
  }
  LODWORD(v24) = v8;
  if ( 4 * v9 )
    memset(v7, 255, 4 * v9);
  v10 = (unsigned int)(1 << a3);
  v11 = a2 ^ 1;
  if ( (_DWORD)v10 )
  {
    v12 = 4 * v10;
    v13 = 0;
    do
    {
      *(_DWORD *)((char *)s1 + v13) = v11;
      v13 += 4;
      v11 += 2;
    }
    while ( v12 != v13 );
  }
  v14 = *(a1 - 3);
  v21 = 0x1000000000LL;
  s2 = v22;
  sub_15FAA20(v14, &s2);
  if ( (unsigned int)v24 == (unsigned __int64)(unsigned int)v21 )
  {
    v18 = 4LL * (unsigned int)v24;
    v15 = s2;
    v16 = 1;
    if ( v18 )
      LOBYTE(v16) = memcmp(s1, s2, v18) == 0;
  }
  else
  {
    v15 = s2;
    v16 = 0;
  }
  if ( v15 != v22 )
    _libc_free((unsigned __int64)v15);
  if ( s1 != s )
    _libc_free((unsigned __int64)s1);
  return v16;
}
