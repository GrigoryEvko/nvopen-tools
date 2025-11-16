// Function: sub_3374700
// Address: 0x3374700
//
__int64 __fastcall sub_3374700(__int64 a1, __int64 a2, int a3, int a4, int a5, __int64 a6, int a7)
{
  int v7; // eax
  __int64 v8; // rdi

  v7 = *(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a1 + 864);
  if ( v7 == 15 || v7 == 39 )
    return sub_33E6410(v8, a4, a5, *(_DWORD *)(a2 + 96), 0, a6, a7);
  else
    return sub_33E60C0(v8, a4, a5, a2, a3, 0, a6, a7);
}
