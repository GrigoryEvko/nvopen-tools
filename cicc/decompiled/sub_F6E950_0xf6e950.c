// Function: sub_F6E950
// Address: 0xf6e950
//
__int64 __fastcall sub_F6E950(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 result; // rax

  v8 = (unsigned int)sub_D4A290(a1, "llvm.loop.licm_versioning.disable", 0x21u, a4, a5, a6);
  result = 6;
  if ( !(_BYTE)v8 )
    return 2
         * (unsigned int)((unsigned __int8)sub_F6E590(a1, (__int64)"llvm.loop.licm_versioning.disable", v6, v7, v8, v9) != 0);
  return result;
}
