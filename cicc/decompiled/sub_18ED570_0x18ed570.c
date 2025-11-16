// Function: sub_18ED570
// Address: 0x18ed570
//
__int64 __fastcall sub_18ED570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  unsigned int v11; // r13d

  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a5;
  *(_QWORD *)(a1 + 24) = a6;
  sub_18EC920(a1, a2);
  if ( *(_QWORD *)(a1 + 32) == *(_QWORD *)(a1 + 40) )
    return 0;
  sub_18E93F0(a1, a2, v6, v7, v8, v9);
  if ( !*(_DWORD *)(a1 + 144) )
    return 0;
  v11 = sub_18ED160(a1);
  sub_18E87D0(a1);
  return v11;
}
