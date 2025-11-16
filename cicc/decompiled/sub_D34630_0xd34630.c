// Function: sub_D34630
// Address: 0xd34630
//
bool __fastcall sub_D34630(__int64 *a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rdi
  bool result; // al
  int v5; // ecx

  v3 = *a1;
  result = 0;
  v5 = *(_DWORD *)(v3 + 4LL * a2);
  if ( v5 != -1 )
    return *(_DWORD *)(v3 + 4LL * a3) == v5;
  return result;
}
