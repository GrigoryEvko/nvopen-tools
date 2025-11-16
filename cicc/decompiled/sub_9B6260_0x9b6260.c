// Function: sub_9B6260
// Address: 0x9b6260
//
__int64 __fastcall sub_9B6260(__int64 a1, const __m128i *a2, unsigned int a3)
{
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  __int64 result; // rax
  unsigned __int8 v8; // [rsp+Fh] [rbp-31h]
  __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-28h]

  v4 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v4 + 8) == 17 )
  {
    v5 = *(_DWORD *)(v4 + 32);
    v10 = v5;
    if ( v5 > 0x40 )
    {
      sub_C43690(&v9, -1, 1);
    }
    else
    {
      v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
      if ( !v5 )
        v6 = 0;
      v9 = v6;
    }
  }
  else
  {
    v10 = 1;
    v9 = 1;
  }
  result = sub_9A6530(a1, (__int64)&v9, a2, a3);
  if ( v10 > 0x40 )
  {
    if ( v9 )
    {
      v8 = result;
      j_j___libc_free_0_0(v9);
      return v8;
    }
  }
  return result;
}
