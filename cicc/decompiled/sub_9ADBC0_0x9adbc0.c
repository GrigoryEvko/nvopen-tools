// Function: sub_9ADBC0
// Address: 0x9adbc0
//
__int64 *__fastcall sub_9ADBC0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, __m128i *a6)
{
  __int64 v9; // rax
  unsigned int v10; // eax
  unsigned __int64 v11; // rdx
  __m128i *v13; // [rsp+0h] [rbp-50h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(v9 + 8) == 17 )
  {
    v10 = *(_DWORD *)(v9 + 32);
    v15 = v10;
    if ( v10 > 0x40 )
    {
      v13 = a6;
      sub_C43690(&v14, -1, 1);
      a6 = v13;
    }
    else
    {
      v11 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v10;
      if ( !v10 )
        v11 = 0;
      v14 = v11;
    }
  }
  else
  {
    v15 = 1;
    v14 = 1;
  }
  sub_9ACEA0(a1, (unsigned __int8 *)a2, (__int64)&v14, a3, a4, a5, a6);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return a1;
}
