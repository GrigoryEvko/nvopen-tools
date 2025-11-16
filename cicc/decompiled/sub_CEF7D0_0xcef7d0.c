// Function: sub_CEF7D0
// Address: 0xcef7d0
//
__int64 __fastcall sub_CEF7D0(int a1, __int64 a2)
{
  size_t v2; // rdx
  char *v3; // r13

  switch ( a1 )
  {
    case 2:
      v3 = off_4C5D0C0[0];
      if ( off_4C5D0C0[0] )
        goto LABEL_4;
      break;
    case 3:
      v2 = 0;
      v3 = off_4C5D0B8[0];
      if ( !off_4C5D0B8[0] )
        return sub_B2D620(a2, v3, v2);
LABEL_4:
      v2 = strlen(v3);
      return sub_B2D620(a2, v3, v2);
    case 1:
      v3 = off_4C5D0C8[0];
      if ( off_4C5D0C8[0] )
        goto LABEL_4;
      break;
    default:
      BUG();
  }
  return sub_B2D620(a2, 0, 0);
}
