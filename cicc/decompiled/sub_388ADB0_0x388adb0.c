// Function: sub_388ADB0
// Address: 0x388adb0
//
__int64 __fastcall sub_388ADB0(__int64 a1, _DWORD *a2)
{
  int v2; // eax

  v2 = *(_DWORD *)(a1 + 64);
  if ( v2 == 40 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    *a2 = 2;
    return 0;
  }
  else if ( v2 == 41 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    *a2 = 1;
    return 0;
  }
  else
  {
    *a2 = 0;
    return 0;
  }
}
