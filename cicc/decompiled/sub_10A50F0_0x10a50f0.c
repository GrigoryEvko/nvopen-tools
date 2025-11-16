// Function: sub_10A50F0
// Address: 0x10a50f0
//
__int64 __fastcall sub_10A50F0(_QWORD **a1, _BYTE *a2)
{
  __int64 result; // rax
  unsigned __int8 *v3; // rcx
  int v4; // edx
  int v5; // edx
  unsigned __int8 *v6; // rcx

  result = 0;
  if ( *a2 == 67 )
  {
    v3 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
    v4 = *v3;
    if ( (unsigned __int8)v4 > 0x1Cu )
    {
      v5 = v4 - 29;
    }
    else
    {
      if ( (_BYTE)v4 != 5 )
        return result;
      v5 = *((unsigned __int16 *)v3 + 1);
    }
    result = 0;
    if ( v5 == 47 )
    {
      if ( (v3[7] & 0x40) != 0 )
        v6 = (unsigned __int8 *)*((_QWORD *)v3 - 1);
      else
        v6 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
      result = 0;
      if ( *(_QWORD *)v6 )
      {
        **a1 = *(_QWORD *)v6;
        return 1;
      }
    }
  }
  return result;
}
