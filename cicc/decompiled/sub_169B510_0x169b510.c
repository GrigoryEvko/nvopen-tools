// Function: sub_169B510
// Address: 0x169b510
//
__int64 __fastcall sub_169B510(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  if ( a3 == 3 )
  {
    if ( *(_WORD *)a2 != 28265 || *(_BYTE *)(a2 + 2) != 102 )
    {
      if ( *(_WORD *)a2 != 24942 || *(_BYTE *)(a2 + 2) != 110 )
      {
        if ( *(_WORD *)a2 != 24910 )
          return 0;
        if ( *(_BYTE *)(a2 + 2) != 78 )
          return 0;
      }
      sub_16986F0(a1, 0, 0, 0);
      return 1;
    }
    goto LABEL_7;
  }
  if ( a3 == 8 )
  {
    if ( *(_QWORD *)a2 != 0x5954494E49464E49LL )
      return 0;
    goto LABEL_7;
  }
  if ( a3 != 4 )
  {
    result = 0;
    if ( a3 != 9 )
      return result;
    if ( *(_QWORD *)a2 == 0x54494E49464E492DLL && *(_BYTE *)(a2 + 8) == 89 )
    {
LABEL_19:
      sub_169B4C0((__int64)a1, 1);
      return 1;
    }
    return 0;
  }
  if ( *(_DWORD *)a2 == 1718503723 )
  {
LABEL_7:
    sub_169B4C0((__int64)a1, 0);
    return 1;
  }
  if ( *(_DWORD *)a2 == 1718511917 || *(_DWORD *)a2 == 1718503725 )
    goto LABEL_19;
  if ( *(_DWORD *)a2 == 1851878957 || (result = 0, *(_DWORD *)a2 == 1314999853) )
  {
    sub_16986F0(a1, 0, 1, 0);
    return 1;
  }
  return result;
}
