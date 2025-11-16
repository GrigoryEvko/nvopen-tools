// Function: sub_17ADB80
// Address: 0x17adb80
//
void __fastcall sub_17ADB80(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  int v4; // r9d
  __int64 i; // rax
  int v6; // edx

  *a3 = 1;
  *a2 = 1;
  v4 = *(_DWORD *)(a1 + 8);
  if ( v4 )
  {
    for ( i = 0; i != v4; ++i )
    {
      v6 = *(_DWORD *)(*(_QWORD *)a1 + 4 * i);
      if ( v6 >= 0 )
      {
        *a2 &= v6 == (_DWORD)i;
        *a3 &= *(_DWORD *)(*(_QWORD *)a1 + 4 * i) - v4 == (_DWORD)i;
      }
    }
  }
}
