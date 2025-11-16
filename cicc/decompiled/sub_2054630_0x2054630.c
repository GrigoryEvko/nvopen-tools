// Function: sub_2054630
// Address: 0x2054630
//
void __fastcall sub_2054630(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13

  if ( a2 )
  {
    v4 = *(_QWORD *)(a1 + 552);
    nullsub_686();
    *(_QWORD *)(v4 + 176) = a2;
    *(_DWORD *)(v4 + 184) = a3;
    sub_1D23870();
  }
  else
  {
    *(_BYTE *)(a1 + 760) = 1;
  }
}
