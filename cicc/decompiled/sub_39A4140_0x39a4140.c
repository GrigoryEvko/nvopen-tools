// Function: sub_39A4140
// Address: 0x39a4140
//
__int64 __fastcall sub_39A4140(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  void *v6; // rax
  size_t v7; // rdx
  __int64 result; // rax
  __int64 *v9; // r12
  unsigned __int64 v10; // r8
  __int64 v11; // r8
  _DWORD v12[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)(a3 + 8 * (2LL - *(unsigned int *)(a3 + 8)));
  if ( v5 )
  {
    v6 = (void *)sub_161E970(v5);
    if ( v7 )
      sub_39A3F30(a1, a2, 3, v6, v7);
  }
  result = *(unsigned __int16 *)(a3 + 2);
  if ( (_DWORD)result != 59 )
  {
    v9 = (__int64 *)(a2 + 8);
    if ( (_DWORD)result != 18 )
    {
      v11 = *(unsigned int *)(a3 + 52);
      v12[0] = 65547;
      sub_39A3560((__int64)a1, v9, 62, (__int64)v12, v11);
    }
    v10 = *(_QWORD *)(a3 + 32);
    BYTE2(v12[0]) = 0;
    return sub_39A3560((__int64)a1, v9, 11, (__int64)v12, v10 >> 3);
  }
  return result;
}
