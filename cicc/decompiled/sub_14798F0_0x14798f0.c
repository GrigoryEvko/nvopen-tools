// Function: sub_14798F0
// Address: 0x14798f0
//
__int64 __fastcall sub_14798F0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        _QWORD *a7,
        __int64 *a8)
{
  unsigned int v8; // r14d
  __int64 v9; // r13
  __int64 result; // rax
  __int64 v13; // rax
  unsigned int v14; // edx
  bool v16; // [rsp+1Fh] [rbp-31h] BYREF

  v8 = a2;
  v9 = a4;
  if ( !sub_146CEE0(a1, a4, a5) )
  {
    if ( !sub_146CEE0(a1, a3, a5) )
      return 0;
    v8 = sub_15FF5D0(a2);
    v13 = a3;
    a3 = v9;
    v9 = v13;
  }
  if ( *(_WORD *)(a3 + 24) != 7 || a5 != *(_QWORD *)(a3 + 48) || !(unsigned __int8)sub_14798E0(a1, a3, v8, &v16) )
    return 0;
  v14 = v8;
  if ( !v16 )
    v14 = sub_15FF0F0(v8);
  result = sub_1474350(a1, a5, v14, a3, v9);
  if ( !(_BYTE)result )
    return 0;
  *a6 = v8;
  *a7 = **(_QWORD **)(a3 + 32);
  *a8 = v9;
  return result;
}
