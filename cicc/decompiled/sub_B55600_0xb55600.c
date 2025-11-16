// Function: sub_B55600
// Address: 0xb55600
//
__int64 __fastcall sub_B55600(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r12
  char v6[32]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v7; // [rsp+20h] [rbp-30h]

  v1 = *(_QWORD *)(a1 - 32);
  v2 = *(_QWORD *)(a1 + 8);
  v7 = 257;
  v3 = sub_BD2C40(72, unk_3F10A14);
  v4 = v3;
  if ( v3 )
    sub_B51B50(v3, v1, v2, (__int64)v6, 0, 0);
  return v4;
}
