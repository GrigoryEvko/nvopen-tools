// Function: sub_9AC0E0
// Address: 0x9ac0e0
//
__int64 __fastcall sub_9AC0E0(__int64 a1, unsigned __int64 *a2, unsigned int a3, __m128i *a4)
{
  __m128i *v4; // r8
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 result; // rax
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-28h]

  v4 = a4;
  v6 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v6 + 8) == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    v12 = v7;
    if ( v7 > 0x40 )
    {
      sub_C43690(&v11, -1, 1);
      v4 = a4;
    }
    else
    {
      v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v7;
      if ( !v7 )
        v8 = 0;
      v11 = v8;
    }
  }
  else
  {
    v12 = 1;
    v11 = 1;
  }
  result = sub_9AB8E0((unsigned __int8 *)a1, (__int64)&v11, a2, a3, v4);
  if ( v12 > 0x40 )
  {
    if ( v11 )
      return j_j___libc_free_0_0(v11);
  }
  return result;
}
