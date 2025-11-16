// Function: sub_F6E5D0
// Address: 0xf6e5d0
//
__int64 __fastcall sub_F6E5D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // [rsp+8h] [rbp-18h]

  if ( (unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll.disable", 0x18u, a4, a5, a6) )
    return 6;
  v20 = sub_D4A2B0(a1, "llvm.loop.unroll.count", 0x16u, v6, v7, v8);
  if ( BYTE4(v20) )
  {
    if ( (_DWORD)v20 != 1 )
      return 5;
    return 6;
  }
  if ( (unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll.enable", 0x17u, v9, v10, v11)
    || (unsigned __int8)sub_D4A290(a1, "llvm.loop.unroll.full", 0x15u, v13, v14, v15) )
  {
    return 5;
  }
  return 2 * (unsigned int)((unsigned __int8)sub_F6E590(a1, (__int64)"llvm.loop.unroll.full", v16, v17, v18, v19) != 0);
}
