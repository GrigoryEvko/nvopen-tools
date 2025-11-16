// Function: sub_95E3E0
// Address: 0x95e3e0
//
__int64 __fastcall sub_95E3E0(__int64 a1, __m128i *a2, size_t a3, int *a4, __m128i *a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  char *v8; // r13
  __int64 *v10; // rax
  char *v11; // rbx
  __int64 v12; // [rsp+0h] [rbp-78h]
  char *v13; // [rsp+8h] [rbp-70h] BYREF
  unsigned int v14; // [rsp+10h] [rbp-68h]
  char v15; // [rsp+18h] [rbp-60h] BYREF

  sub_95D990(&v13, a2, a3, a4, a5, 0, 61);
  v5 = v14;
  v6 = a1 + 16;
  v7 = v12;
  if ( v14 )
  {
    *(_QWORD *)a1 = v6;
    v10 = (__int64 *)&v13[32 * v5 - 32];
    v7 = *v10;
    sub_95BD60((__int64 *)a1, (_BYTE *)*v10, *v10 + v10[1]);
    v8 = v13;
    v11 = &v13[32 * v14];
    if ( v13 != v11 )
    {
      do
      {
        v11 -= 32;
        if ( *(char **)v11 != v11 + 16 )
        {
          v7 = *((_QWORD *)v11 + 2) + 1LL;
          j_j___libc_free_0(*(_QWORD *)v11, v7);
        }
      }
      while ( v8 != v11 );
      v8 = v13;
    }
  }
  else
  {
    *(_QWORD *)a1 = v6;
    v8 = v13;
    *(_QWORD *)(a1 + 8) = 0;
    *(_BYTE *)(a1 + 16) = 0;
  }
  if ( v8 != &v15 )
    _libc_free(v8, v7);
  return a1;
}
