// Function: sub_2E45450
// Address: 0x2e45450
//
__int64 __fastcall sub_2E45450(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v7; // r14d
  __int64 *v8; // rax
  unsigned int v9; // r8d
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD v13[8]; // [rsp+0h] [rbp-40h] BYREF

  sub_2E44C10((__int64)v13, a2, *(_QWORD *)(a1 + 8), *(_BYTE *)(a1 + 24));
  v7 = *(_DWORD *)(v13[0] + 8LL);
  v8 = (__int64 *)sub_2E8A250(a3, a4, *(_QWORD *)(a1 + 8), *(_QWORD *)a1);
  v9 = 0;
  if ( v8 )
  {
    if ( v7 - 1 <= 0x3FFFFFFE )
    {
      v10 = *v8;
      v11 = v7 >> 3;
      if ( (unsigned int)v11 < *(unsigned __int16 *)(v10 + 22) )
        return ((int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + v11) >> (v7 & 7)) & 1;
    }
  }
  return v9;
}
