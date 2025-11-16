// Function: sub_68AF60
// Address: 0x68af60
//
_BOOL8 __fastcall sub_68AF60(__int64 **a1, __int64 a2)
{
  _BOOL8 result; // rax
  __int64 *v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // rdx

  result = 0;
  if ( (*(_BYTE *)(a2 + 89) & 1) != 0 )
  {
    v3 = *a1;
    if ( *a1 )
    {
      while ( 1 )
      {
        v4 = *((_BYTE *)v3 + 32);
        if ( (v4 & 3) == 0 && a2 == v3[1] )
          return ((v4 >> 3) ^ 1) & 1;
        v3 = (__int64 *)*v3;
        if ( !v3 )
          goto LABEL_6;
      }
    }
    else
    {
LABEL_6:
      v5 = *(_QWORD *)(a2 + 40);
      result = 0;
      if ( v5 )
      {
        if ( *(_DWORD *)(v5 + 240) < dword_4F04C58 )
          return ((_BYTE)a1[3] & 0x30) == 16;
      }
    }
  }
  return result;
}
