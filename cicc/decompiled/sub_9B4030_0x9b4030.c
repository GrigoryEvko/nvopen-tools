// Function: sub_9B4030
// Address: 0x9b4030
//
__int64 __fastcall sub_9B4030(__int64 *a1, int a2, unsigned int a3, __m128i *a4)
{
  unsigned int v4; // r8d
  __m128i *v5; // r9
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 v12; // [rsp+18h] [rbp-38h] BYREF
  unsigned __int64 v13; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-28h]

  v4 = a3;
  v5 = a4;
  v6 = a1[1];
  v12 = 1023;
  if ( *(_BYTE *)(v6 + 8) == 17 )
  {
    v7 = *(_DWORD *)(v6 + 32);
    v14 = v7;
    if ( v7 > 0x40 )
    {
      sub_C43690(&v13, -1, 1);
      v4 = a3;
      v5 = a4;
    }
    else
    {
      v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v7;
      if ( !v7 )
        v8 = 0;
      v13 = v8;
    }
  }
  else
  {
    v14 = 1;
    v13 = 1;
  }
  sub_9B0780(a1, (__int64 *)&v13, a2, (unsigned int *)&v12, v4, v5);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  return v12;
}
