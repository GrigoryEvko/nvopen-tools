// Function: sub_C2F370
// Address: 0xc2f370
//
__int64 __fastcall sub_C2F370(__int64 a1)
{
  __int64 v1; // rdx

  v1 = *(_QWORD *)(a1 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v1) <= 6 )
  {
    sub_CB6200(a1, "REMARKS", 7);
  }
  else
  {
    *(_DWORD *)v1 = 1095583058;
    *(_WORD *)(v1 + 4) = 19282;
    *(_BYTE *)(v1 + 6) = 83;
    *(_QWORD *)(a1 + 32) += 7LL;
  }
  return sub_CB5D20(a1, 0);
}
