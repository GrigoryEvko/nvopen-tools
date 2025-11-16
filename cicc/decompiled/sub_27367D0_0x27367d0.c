// Function: sub_27367D0
// Address: 0x27367d0
//
char __fastcall sub_27367D0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned int v3; // eax
  int v5; // r15d
  unsigned int i; // ebx

  v3 = *a3 - 67;
  if ( v3 > 0xC )
  {
    v5 = *((_DWORD *)a3 + 1) & 0x7FFFFFF;
    if ( v5 )
    {
      for ( i = 0; i != v5; ++i )
      {
        while ( 1 )
        {
          LOBYTE(v3) = sub_F58730(a3, i);
          if ( (_BYTE)v3 )
            break;
          if ( v5 == ++i )
            return v3;
        }
        LOBYTE(v3) = sub_27366A0(a1, a2, (__int64)a3, i);
      }
    }
  }
  return v3;
}
