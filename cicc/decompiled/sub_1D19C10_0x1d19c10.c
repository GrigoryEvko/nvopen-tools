// Function: sub_1D19C10
// Address: 0x1d19c10
//
__int64 __fastcall sub_1D19C10(__int64 a1)
{
  _QWORD *v1; // rax

  v1 = *(_QWORD **)(a1 + 88);
  if ( *(int *)(a1 + 96) < 0 )
    return v1[1];
  else
    return *v1;
}
