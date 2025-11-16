// Function: sub_1E0AB50
// Address: 0x1e0ab50
//
__int64 __fastcall sub_1E0AB50(__int64 a1)
{
  _QWORD *v1; // rax

  v1 = *(_QWORD **)a1;
  if ( *(int *)(a1 + 8) < 0 )
    return v1[1];
  else
    return *v1;
}
