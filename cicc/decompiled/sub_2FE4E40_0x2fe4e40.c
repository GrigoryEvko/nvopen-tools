// Function: sub_2FE4E40
// Address: 0x2fe4e40
//
__int64 __fastcall sub_2FE4E40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int); // rax
  _DWORD *v5; // rax
  unsigned __int16 v6; // dx
  int v7; // eax

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  if ( v4 != sub_2D42F30 )
    return ((unsigned __int16 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v4)(a1, a2, 0, a4, a2);
  v5 = sub_AE2980(a2, 0);
  v6 = 2;
  v7 = v5[1];
  if ( v7 != 1 )
  {
    v6 = 3;
    if ( v7 != 2 )
    {
      v6 = 4;
      if ( v7 != 4 )
      {
        v6 = 5;
        if ( v7 != 8 )
        {
          v6 = 6;
          if ( v7 != 16 )
          {
            v6 = 7;
            if ( v7 != 32 )
            {
              v6 = 8;
              if ( v7 != 64 )
                return (unsigned __int16)(9 * (v7 == 128));
            }
          }
        }
      }
    }
  }
  return v6;
}
