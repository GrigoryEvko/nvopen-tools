// Function: sub_40ECF5
// Address: 0x40ecf5
//
__int64 __fastcall sub_40ECF5(int a1, __int64 *a2, __int64 a3, int a4, int a5, int a6, char a7)
{
  __int64 v7; // rbx

  v7 = *a2;
  do
  {
    if ( !v7 )
      break;
    sub_40EBBB(a1, *(_DWORD *)v7, *(_DWORD *)(v7 + 4), *(_DWORD *)(v7 + 8), (const char **)(v7 + 16));
    v7 = *(_QWORD *)(v7 + 24);
  }
  while ( v7 != *a2 );
  return sub_130F1C0(a1, (unsigned int)"\n", a4, a4, a5, a6, a7);
}
