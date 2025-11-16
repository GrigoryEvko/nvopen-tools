// Function: sub_86A660
// Address: 0x86a660
//
__int64 __fastcall sub_86A660(_QWORD **a1)
{
  char v1; // al
  __int64 v2; // r13
  _QWORD *v4; // rdx
  char v5; // al
  _QWORD *v6; // rax
  char v7; // dl
  _BOOL4 v8; // eax

  v1 = *((_BYTE *)a1 + 16);
  if ( v1 != 6 )
  {
    switch ( v1 )
    {
      case 7:
        v8 = (*((_BYTE *)a1[3] + 175) & 8) != 0;
        break;
      case 53:
        v6 = a1[3];
        v7 = *((_BYTE *)v6 + 16);
        if ( v7 != 7 && v7 != 11 )
          goto LABEL_5;
        v8 = (*((_BYTE *)v6 + 57) & 2) != 0;
        break;
      case 11:
        v8 = (*((_BYTE *)a1[3] + 206) & 0x40) != 0;
        break;
      default:
LABEL_5:
        v2 = (__int64)*a1;
        sub_86A080(a1);
        return v2;
    }
    if ( v8 )
      sub_86A530(*a1);
    goto LABEL_5;
  }
  v4 = a1[3];
  v5 = *((_BYTE *)v4 + 140);
  if ( (unsigned __int8)(v5 - 9) > 2u && (v5 != 2 || (*((_BYTE *)v4 + 161) & 8) == 0) )
    goto LABEL_5;
  return sub_869630((__int64)a1, 0);
}
