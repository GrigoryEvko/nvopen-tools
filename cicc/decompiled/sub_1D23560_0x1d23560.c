// Function: sub_1D23560
// Address: 0x1d23560
//
__int64 __fastcall sub_1D23560(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // esi
  __int64 v4; // rdx
  int v5; // ecx
  unsigned int v6; // eax
  int v7; // edx
  _QWORD v9[3]; // [rsp+0h] [rbp-20h] BYREF

  v9[0] = a2;
  v9[1] = a3;
  if ( (_BYTE)a2 )
    v3 = word_42E7700[(unsigned __int8)(a2 - 14)];
  else
    v3 = sub_1F58D30(v9);
  if ( v3 )
  {
    v4 = 0;
    while ( 1 )
    {
      v5 = *(_DWORD *)(a1 + 4 * v4);
      v6 = v4;
      if ( v5 >= 0 )
        break;
      if ( v3 == ++v4 )
        return 1;
    }
    if ( v3 != (_DWORD)v4 )
    {
      while ( ++v6 != v3 )
      {
        v7 = *(_DWORD *)(a1 + 4LL * v6);
        if ( v5 != v7 && v7 >= 0 )
          return 0;
      }
    }
  }
  return 1;
}
