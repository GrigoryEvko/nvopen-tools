// Function: sub_21CEE40
// Address: 0x21cee40
//
__int64 __fastcall sub_21CEE40(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  unsigned int v8; // r13d
  __int64 v9; // rax
  _QWORD v10[6]; // [rsp+0h] [rbp-30h] BYREF

  v10[0] = a4;
  v10[1] = a5;
  if ( (_BYTE)a4 )
  {
    if ( (unsigned __int8)(a4 - 14) > 0x5Fu )
      return 2;
    v8 = (unsigned __int16)word_435D740[(unsigned __int8)(a4 - 14)];
  }
  else
  {
    if ( !sub_1F58D20((__int64)v10) )
      return 2;
    v8 = sub_1F58D30((__int64)v10);
  }
  LOBYTE(v9) = sub_1D15020(2, v8);
  if ( !(_BYTE)v9 )
  {
    v9 = sub_1F593D0(a3, 2, 0, v8);
    v5 = v9;
  }
  LOBYTE(v5) = v9;
  return v5;
}
