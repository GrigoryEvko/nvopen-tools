// Function: sub_16BDFB0
// Address: 0x16bdfb0
//
void __fastcall sub_16BDFB0(int *a1, char *a2, __int64 a3)
{
  char *v3; // r8
  int v4; // edx
  int v5; // ecx
  char v6; // al

  v3 = &a2[a3];
  if ( a2 != &a2[a3] )
  {
    while ( 1 )
    {
      v4 = *a1;
      v5 = *a1 + 1;
      *a1 = v5;
      v6 = *a2;
      if ( *a2 == 10 )
        break;
      if ( v6 == 13 )
      {
        *a1 = 0;
LABEL_9:
        if ( v3 == ++a2 )
          return;
      }
      else
      {
        if ( v6 == 9 )
          *a1 = v5 + (~(_BYTE)v4 & 7);
        if ( v3 == ++a2 )
          return;
      }
    }
    ++a1[1];
    *a1 = 0;
    goto LABEL_9;
  }
}
