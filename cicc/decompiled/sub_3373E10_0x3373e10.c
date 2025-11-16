// Function: sub_3373E10
// Address: 0x3373e10
//
__int64 __fastcall sub_3373E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  if ( !*(_QWORD *)(*(_QWORD *)(a1 + 960) + 32LL) )
    return sub_2E321B0(a2, a3, a3, a4, a5, a6);
  if ( (_DWORD)a4 != -1 )
    return sub_2E33F80(a2, a3, a4, a4, a5, a6);
  v8 = sub_3373D80(a1, a2, a3);
  return sub_2E33F80(a2, a3, v8, v9, v10, v11);
}
