// Function: sub_15A5A00
// Address: 0x15a5a00
//
__int64 __fastcall sub_15A5A00(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  int v5; // r10d
  __int64 v8; // r13

  v5 = 0;
  v8 = *(_QWORD *)(a1 + 8);
  if ( a3 )
    v5 = sub_161FF10(*(_QWORD *)(a1 + 8), a2, a3);
  return sub_15BC830(v8, 36, v5, a4, 0, a5, 0, 1);
}
