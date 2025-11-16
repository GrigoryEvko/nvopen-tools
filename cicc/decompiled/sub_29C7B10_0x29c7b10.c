// Function: sub_29C7B10
// Address: 0x29c7b10
//
__int64 __fastcall sub_29C7B10(__int64 a1, int a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 *v7; // rdi
  _BYTE v8[16]; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v9)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-20h]

  if ( a2 != 1 )
    return (unsigned int)sub_29C70F0(
                           *(_QWORD *)(a1 + 40),
                           *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL),
                           *(_QWORD *)(a1 + 40) + 24LL,
                           a3,
                           "FunctionDebugify (original debuginfo)",
                           0x25u);
  v5 = *(_QWORD *)(a1 + 64);
  v6 = a1 + 56;
  v7 = *(__int64 **)(a1 + 40);
  v9 = 0;
  v3 = sub_29C2F90(v7, v6, v5, "FunctionDebugify: ", 0x12u, (__int64)v8);
  if ( !v9 )
    return v3;
  v9(v8, v8, 3);
  return v3;
}
