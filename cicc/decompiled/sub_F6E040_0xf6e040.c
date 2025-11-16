// Function: sub_F6E040
// Address: 0xf6e040
//
__int64 __fastcall sub_F6E040(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  bool v10; // al
  __int64 v11; // [rsp+14h] [rbp-2Ch]
  __int64 v12; // [rsp+1Ch] [rbp-24h]
  __int64 v13; // [rsp+24h] [rbp-1Ch]

  v11 = sub_D4A2B0(a1, "llvm.loop.vectorize.width", 0x19u, a4, a5, a6);
  if ( !BYTE4(v11) )
    return v13;
  v12 = sub_D4A2B0(a1, "llvm.loop.vectorize.scalable.enable", 0x23u, v6, v7, v8);
  v10 = 0;
  if ( BYTE4(v12) )
    v10 = (_DWORD)v12 != 0;
  BYTE4(v13) = v10;
  LODWORD(v13) = v11;
  return v13;
}
