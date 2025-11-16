// Function: sub_15AB790
// Address: 0x15ab790
//
void __fastcall sub_15AB790(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // al
  int v3; // edx

  if ( a2 )
  {
    while ( 1 )
    {
      v2 = *a2;
      if ( *a2 > 0xEu )
      {
        if ( (unsigned __int8)(v2 - 32) <= 1u )
        {
LABEL_11:
          sub_15ABBA0();
          return;
        }
        if ( v2 == 16 )
        {
          sub_15AB320(a1, (__int64)a2);
          return;
        }
        if ( v2 == 17 )
        {
          sub_15ABAC0();
          return;
        }
      }
      else if ( v2 > 0xAu )
      {
        goto LABEL_11;
      }
      if ( !(unsigned __int8)sub_15AB650(a1, (__int64)a2) )
        return;
      v3 = *a2;
      if ( (unsigned int)(v3 - 18) <= 1 || (_BYTE)v3 == 20 )
      {
        a2 = *(unsigned __int8 **)&a2[8 * (1LL - *((unsigned int *)a2 + 2))];
        if ( !a2 )
          return;
      }
      else
      {
        if ( (_BYTE)v3 != 21 )
          return;
        a2 = *(unsigned __int8 **)&a2[-8 * *((unsigned int *)a2 + 2)];
        if ( !a2 )
          return;
      }
    }
  }
}
