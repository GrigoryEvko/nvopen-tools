// Function: sub_CEF870
// Address: 0xcef870
//
__int64 __fastcall sub_CEF870(int a1, __int64 a2)
{
  size_t v2; // rdx
  char *v3; // r13

  switch ( a1 )
  {
    case 2:
      v2 = 0;
      v3 = off_4C5D0C0[0];
      if ( !off_4C5D0C0[0] )
        return sub_B2CD60(a2, v3, v2, 0, 0);
      goto LABEL_4;
    case 3:
      v2 = 0;
      v3 = off_4C5D0B8[0];
      if ( off_4C5D0B8[0] )
LABEL_4:
        v2 = strlen(v3);
      break;
    case 1:
      v2 = 0;
      v3 = off_4C5D0C8[0];
      if ( off_4C5D0C8[0] )
        goto LABEL_4;
      break;
    default:
      BUG();
  }
  return sub_B2CD60(a2, v3, v2, 0, 0);
}
