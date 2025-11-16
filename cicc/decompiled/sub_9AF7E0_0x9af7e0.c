// Function: sub_9AF7E0
// Address: 0x9af7e0
//
__int64 __fastcall sub_9AF7E0(__int64 a1, unsigned int a2, __m128i *a3)
{
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  __int64 result; // rax
  unsigned int v8; // [rsp+Ch] [rbp-34h]
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
  result = sub_9AE470(a1, (__int64)&v9, a2, a3);
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
