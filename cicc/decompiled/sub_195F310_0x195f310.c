// Function: sub_195F310
// Address: 0x195f310
//
__int64 __fastcall sub_195F310(__int64 a1, __int64 a2, __int64 a3, char *a4, __int64 *a5)
{
  unsigned int v7; // r12d
  __int64 v8; // rsi
  __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = sub_1437020(a1, a2, a3, a4);
  if ( (_BYTE)v7 )
    return v7;
  if ( *(_BYTE *)(a1 + 16) != 54 )
    return v7;
  v8 = *(_QWORD *)(a1 - 24);
  v10[0] = a1;
  if ( !sub_13FC1A0(a3, v8) )
    return v7;
  sub_195EFD0(a5, v10);
  return v7;
}
