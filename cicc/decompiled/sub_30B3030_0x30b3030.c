// Function: sub_30B3030
// Address: 0x30b3030
//
__int64 __fastcall sub_30B3030(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_22077B0(0x10u);
  v5 = v4;
  if ( v4 )
  {
    *(_QWORD *)v4 = a3;
    *(_DWORD *)(v4 + 8) = 1;
  }
  v7[0] = v4;
  sub_30B2AC0(a2 + 8, v7);
  return v5;
}
