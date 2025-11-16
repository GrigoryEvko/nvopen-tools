// Function: sub_16C9150
// Address: 0x16c9150
//
__int64 __fastcall sub_16C9150(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rdx
  _QWORD v8[3]; // [rsp+10h] [rbp-18h] BYREF

  v8[0] = 0;
  result = sub_16B3750(a1 + 192, a1, a3, a4, a5, a6, v8);
  if ( !(_BYTE)result )
  {
    v7 = v8[0];
    *(_DWORD *)(a1 + 16) = a2;
    *(_QWORD *)(a1 + 160) = v7;
  }
  return result;
}
