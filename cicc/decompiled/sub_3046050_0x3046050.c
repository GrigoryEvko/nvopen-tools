// Function: sub_3046050
// Address: 0x3046050
//
char __fastcall sub_3046050(__int64 a1, __int64 *a2, int a3)
{
  int v3; // eax
  char result; // al

  v3 = *(_DWORD *)(*(_QWORD *)(a1 + 537016) + 352LL);
  if ( v3 != -1 )
    return v3 > 0;
  if ( !a3 )
    return 0;
  result = 1;
  if ( *(_DWORD *)(a2[1] + 952) )
    return sub_3046000(a1, a2);
  return result;
}
