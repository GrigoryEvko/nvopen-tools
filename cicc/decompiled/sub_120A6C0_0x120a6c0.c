// Function: sub_120A6C0
// Address: 0x120a6c0
//
__int64 __fastcall sub_120A6C0(__int64 a1, _DWORD *a2)
{
  int v2; // eax

  v2 = *(_DWORD *)(a1 + 240);
  if ( v2 == 42 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    *a2 = 2;
    return 0;
  }
  else if ( v2 == 43 )
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    *a2 = 1;
    return 0;
  }
  else
  {
    *a2 = 0;
    return 0;
  }
}
