// Function: sub_287EE90
// Address: 0x287ee90
//
__int64 __fastcall sub_287EE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int8 v8; // dl
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax

  v6 = sub_D49300(a1, a2, a3, a4, a5, a6);
  if ( !v6 )
    return 0;
  v7 = sub_2A11940(v6, "llvm.loop.unroll.count", 22);
  if ( !v7 )
    return 0;
  v8 = *(_BYTE *)(v7 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(_QWORD *)(v7 - 32);
  else
    v9 = v7 - 8LL * ((v8 >> 2) & 0xF) - 16;
  v10 = *(_QWORD *)(*(_QWORD *)(v9 + 8) + 136LL);
  result = *(_QWORD *)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
