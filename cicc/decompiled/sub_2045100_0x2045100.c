// Function: sub_2045100
// Address: 0x2045100
//
void __fastcall sub_2045100(__int64 a1, __int64 a2, int a3)
{
  if ( a2 )
  {
    nullsub_686();
    *(_QWORD *)(a1 + 176) = a2;
    *(_DWORD *)(a1 + 184) = a3;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(a1 + 176) = 0;
    *(_DWORD *)(a1 + 184) = a3;
  }
}
