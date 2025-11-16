// Function: sub_127CAE0
// Address: 0x127cae0
//
void __fastcall sub_127CAE0(__int64 a1, __int64 a2)
{
  __int64 *i; // rbx
  unsigned __int64 v3; // rax
  unsigned __int64 v4[5]; // [rsp+8h] [rbp-28h] BYREF

  for ( i = *(__int64 **)(a2 + 72); i; i = (__int64 *)*i )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)i + 8) == 7 )
      {
        v3 = i[2];
        v4[0] = v3;
        if ( (*(_BYTE *)(v3 + 170) & 0x60) == 0 && *(_BYTE *)(v3 + 177) != 5 )
          break;
      }
      i = (__int64 *)*i;
      if ( !i )
        return;
    }
    sub_91CFF0(a1 + 24, v4);
  }
}
