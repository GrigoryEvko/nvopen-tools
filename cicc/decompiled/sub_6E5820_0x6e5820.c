// Function: sub_6E5820
// Address: 0x6e5820
//
void __fastcall sub_6E5820(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r12

  if ( a1 )
  {
    v2 = a1;
    do
    {
      while ( 1 )
      {
        v4 = *v2;
        if ( (*v2 & 0x40) == 0 )
        {
          if ( (v4 & 0x10) != 0 && (a2 & 0x28) != 0 && !*((_BYTE *)v2 + 8) )
          {
            sub_875E10(*v2, v2[2], v2 + 4, 1, v2[3]);
            *((_BYTE *)v2 + 8) = 1;
          }
          if ( (a2 & 0x2000) != 0 && (unsigned int)sub_8809D0(v2[2]) )
          {
            if ( !qword_4D03C50 || (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0 )
              *v2 |= 4u;
            if ( !*((_BYTE *)v2 + 8) )
              sub_875E10(*v2, v2[2], v2 + 4, 1, v2[3]);
          }
          *((_BYTE *)v2 + 8) = 0;
          v3 = a2 | v4 & 0xFFFFFFFFFFFECF87LL;
          *v2 = v3;
          if ( (v3 & 0x20) != 0 )
            break;
        }
        v2 = (unsigned __int64 *)v2[6];
        if ( !v2 )
          return;
      }
      sub_6E5700(v2);
      v2 = (unsigned __int64 *)v2[6];
    }
    while ( v2 );
  }
}
