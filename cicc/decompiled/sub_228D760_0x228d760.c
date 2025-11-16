// Function: sub_228D760
// Address: 0x228d760
//
bool __fastcall sub_228D760(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rdi
  __int64 v4; // r8

  if ( !a3 )
    return 1;
  v3 = *(_QWORD *)(a1 + 8);
  do
  {
    v4 = (__int64)a3;
    a3 = (_QWORD *)*a3;
  }
  while ( a3 );
  return sub_DADE90(v3, a2, v4);
}
