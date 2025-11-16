// Function: sub_2E39560
// Address: 0x2e39560
//
unsigned __int64 __fastcall sub_2E39560(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax

  v2 = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)v2 >= *(_QWORD *)(a2 + 24) )
  {
    sub_CB5D20(a2, 37);
  }
  else
  {
    *(_QWORD *)(a2 + 32) = v2 + 1;
    *v2 = 37;
  }
  return sub_2E37380(a1, a2, 0, 0);
}
