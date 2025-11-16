// Function: sub_15A5960
// Address: 0x15a5960
//
__int64 __fastcall sub_15A5960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  unsigned int v5; // r13d
  __int64 v7; // rcx
  __int64 v8; // r14

  v5 = a5;
  v7 = 0;
  v8 = *(_QWORD *)(a1 + 8);
  if ( a3 )
    v7 = sub_161FF10(*(_QWORD *)(a1 + 8), a2, a3);
  return sub_15BC290(v8, a4, v5, v7, 0, 1);
}
