// Function: sub_88F140
// Address: 0x88f140
//
__int64 __fastcall sub_88F140(__int64 a1, unsigned __int64 a2, unsigned int a3, _DWORD *a4)
{
  int v5; // r8d
  int v6; // eax
  __int64 result; // rax
  int v8; // [rsp+14h] [rbp-24h] BYREF
  __int64 v9; // [rsp+18h] [rbp-20h] BYREF
  _QWORD v10[3]; // [rsp+20h] [rbp-18h] BYREF

  v5 = sub_7C8F90(a2, (unsigned int *)a3, &v8, 0, 0, (__int64)&v9, v10);
  v6 = v8;
  if ( v8 | v5 )
  {
    *(_DWORD *)(a1 + 36) = 1;
    if ( v6 )
      sub_6851C0(0x292u, a4);
  }
  *(_QWORD *)(a1 + 472) = v9;
  result = v10[0];
  *(_QWORD *)(a1 + 480) = v10[0];
  return result;
}
