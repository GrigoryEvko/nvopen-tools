// Function: sub_15DC150
// Address: 0x15dc150
//
__int64 __fastcall sub_15DC150(__int64 a1)
{
  __int64 v1; // rdx

  v1 = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)v1 )
  {
    sub_15DC140(*(_QWORD *)a1, *(__int64 **)(a1 + 8), v1);
    *(_DWORD *)(a1 + 16) = 0;
  }
  sub_15CDC00(a1);
  return *(_QWORD *)a1;
}
