// Function: sub_869F90
// Address: 0x869f90
//
__int64 __fastcall sub_869F90(_QWORD *a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rdx
  __int64 result; // rax

  v1 = qword_4F04C68[0];
  if ( (*(_BYTE *)(a1 - 1) & 1) == 0 )
    v1 = 776LL * dword_4F04C58 + qword_4F04C68[0];
  v2 = (_QWORD *)(v1 + 320);
  result = *(_QWORD *)(v1 + 320);
  *a1 = result;
  *v2 = a1;
  return result;
}
