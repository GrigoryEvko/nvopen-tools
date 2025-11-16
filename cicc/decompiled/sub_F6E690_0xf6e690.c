// Function: sub_F6E690
// Address: 0xf6e690
//
__int64 __fastcall sub_F6E690(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // [rsp+8h] [rbp-18h]

  if ( (unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll_and_jam.disable", 0x20u, a4, a5, a6) )
    return 6;
  v17 = sub_D4A2B0(a1, "llvm.loop.unroll_and_jam.count", 0x1Eu, v6, v7, v8);
  if ( BYTE4(v17) )
  {
    if ( (_DWORD)v17 != 1 )
      return 5;
    return 6;
  }
  if ( (unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll_and_jam.enable", 0x1Fu, v9, v10, v11) )
    return 5;
  return 2
       * (unsigned int)((unsigned __int8)sub_F6E590(a1, (__int64)"llvm.loop.unroll_and_jam.enable", v13, v14, v15, v16) != 0);
}
