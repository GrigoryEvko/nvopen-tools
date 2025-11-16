// Function: sub_33E6440
// Address: 0x33e6440
//
__int64 __fastcall sub_33E6440(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v6; // rsi
  __int64 *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int8 *v10; // rsi
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *a3;
  v12[0] = v6;
  if ( v6 )
    sub_B96E90((__int64)v12, v6, 1);
  v7 = *(__int64 **)(a1 + 720);
  v8 = *v7;
  v7[10] += 24;
  v9 = (v8 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( v7[1] >= (unsigned __int64)(v9 + 24) && v8 )
  {
    *v7 = v9 + 24;
    if ( !v9 )
    {
      v9 = v12[0];
      if ( !v12[0] )
        return v9;
      sub_B91220((__int64)v12, v12[0]);
      return 0;
    }
  }
  else
  {
    v9 = sub_9D1E70((__int64)v7, 24, 24, 4);
  }
  *(_QWORD *)v9 = a2;
  v10 = (unsigned __int8 *)v12[0];
  *(_QWORD *)(v9 + 8) = v12[0];
  if ( v10 )
  {
    sub_B976B0((__int64)v12, v10, v9 + 8);
    *(_DWORD *)(v9 + 16) = a4;
    return v9;
  }
  *(_DWORD *)(v9 + 16) = a4;
  return v9;
}
