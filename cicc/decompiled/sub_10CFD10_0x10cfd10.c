// Function: sub_10CFD10
// Address: 0x10cfd10
//
bool __fastcall sub_10CFD10(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v4; // rax
  __int64 v5; // rcx

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( v4 )
  {
    **(_QWORD **)a1 = v4;
    result = 1;
    v5 = *((_QWORD *)a3 - 4);
    if ( *(_QWORD *)(a1 + 8) == v5 )
      return result;
  }
  else
  {
    v5 = *((_QWORD *)a3 - 4);
  }
  if ( !v5 )
    return 0;
  **(_QWORD **)a1 = v5;
  return *((_QWORD *)a3 - 8) == *(_QWORD *)(a1 + 8);
}
