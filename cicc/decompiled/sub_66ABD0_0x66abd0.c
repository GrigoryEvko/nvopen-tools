// Function: sub_66ABD0
// Address: 0x66abd0
//
void __fastcall sub_66ABD0(__int64 *a1)
{
  int v1; // edx
  __int64 *v2; // rbx
  char v3; // al

  if ( a1 )
  {
    v1 = 0;
    v2 = a1;
    do
    {
      while ( 1 )
      {
        v3 = *((_BYTE *)v2 + 9);
        if ( v3 == 4 || v3 == 1 )
          break;
        v2 = (__int64 *)*v2;
        if ( !v2 )
          return;
      }
      if ( !v1 )
        sub_6851C0(1882, v2[5]);
      *((_BYTE *)v2 + 8) = 0;
      v2 = (__int64 *)*v2;
      v1 = 1;
    }
    while ( v2 );
  }
}
