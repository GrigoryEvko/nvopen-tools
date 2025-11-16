// Function: sub_CA5770
// Address: 0xca5770
//
char __fastcall sub_CA5770(__int64 a1)
{
  int v1; // eax

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 == 1 )
    return 1;
  if ( v1 == 2 )
    return 0;
  if ( v1 )
    BUG();
  return off_4C5C768(*(_QWORD *)a1);
}
