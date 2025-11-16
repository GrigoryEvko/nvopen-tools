// Function: sub_1687040
// Address: 0x1687040
//
int __fastcall sub_1687040(_QWORD *a1)
{
  _QWORD *v2; // r12
  __int64 i; // rbx
  _QWORD *v4; // rdi

  v2 = (_QWORD *)*a1;
  if ( *a1 )
  {
    for ( i = 0; i != 16; ++i )
    {
      while ( 1 )
      {
        if ( !*((_BYTE *)v2 + i + 12) )
        {
          v4 = (_QWORD *)v2[i + 4];
          if ( v4 )
            break;
        }
        if ( ++i == 16 )
          goto LABEL_7;
      }
      sub_1686480(v4);
    }
LABEL_7:
    sub_16856A0(v2);
  }
  return sub_16856A0(a1);
}
