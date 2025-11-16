// Function: sub_11A68C0
// Address: 0x11a68c0
//
_BYTE *__fastcall sub_11A68C0(__int64 *a1, __int64 *a2, unsigned int *a3, __int64 *a4, __int64 a5)
{
  unsigned int v5; // eax
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 v8; // rsi
  _BYTE *result; // rax
  __int64 v10; // [rsp+0h] [rbp-10h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-8h]

  v5 = *((_DWORD *)a4 + 2);
  v6 = *a1;
  *((_DWORD *)a4 + 2) = 0;
  v7 = *a3;
  v11 = v5;
  v8 = *a2;
  v10 = *a4;
  result = sub_11A3D00(v6, v8, v7, (__int64)&v10, a5);
  if ( v11 > 0x40 )
  {
    if ( v10 )
      return (_BYTE *)j_j___libc_free_0_0(v10);
  }
  return result;
}
