// Function: sub_1C08A20
// Address: 0x1c08a20
//
__int64 __fastcall sub_1C08A20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  unsigned int v8; // r14d
  __int64 v9; // r12
  bool v10; // al
  int v12; // [rsp+Ch] [rbp-34h]

  v6 = sub_157EBA0(a2);
  if ( !v6 )
    return 0;
  v7 = v6;
  v12 = sub_15F3BE0(v6);
  if ( !v12 )
    return 0;
  v8 = 0;
  while ( 1 )
  {
    v9 = sub_15F3BF0(v7, v8);
    v10 = sub_15CCCD0(a4, a3, v9);
    if ( v9 == a3 || v10 )
      break;
    if ( v12 == ++v8 )
      return 0;
  }
  return v9;
}
