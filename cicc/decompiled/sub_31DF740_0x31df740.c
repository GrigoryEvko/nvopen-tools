// Function: sub_31DF740
// Address: 0x31df740
//
__int64 __fastcall sub_31DF740(__int64 a1)
{
  char v1; // al

  v1 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 224) + 8LL) + 1906LL);
  if ( !v1 )
    return 4;
  if ( v1 != 1 )
    BUG();
  return 12;
}
