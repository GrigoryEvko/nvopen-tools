// Function: sub_15CCE20
// Address: 0x15cce20
//
char __fastcall sub_15CCE20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v7[6]; // [rsp+0h] [rbp-30h] BYREF

  if ( !sub_15CC510(a1, a3) )
    return 1;
  v4 = *(_QWORD *)(a2 + 40);
  if ( !sub_15CC510(a1, v4) || v4 == a3 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) != 29 )
    return sub_15CC8F0(a1, v4, a3);
  v5 = *(_QWORD *)(a2 - 48);
  v7[0] = v4;
  v7[1] = v5;
  return sub_15CCD40(a1, v7, a3);
}
