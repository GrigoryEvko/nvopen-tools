// Function: sub_AE8080
// Address: 0xae8080
//
void __fastcall sub_AE8080(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // rax
  int v4; // edx
  unsigned __int8 v5; // al
  __int64 v6; // rsi

  if ( a2 )
  {
    v2 = 0x140000F000LL;
    while ( 1 )
    {
      v3 = *a2;
      if ( (unsigned __int8)v3 <= 0x24u )
      {
        if ( _bittest64(&v2, v3) )
          break;
      }
      if ( (_BYTE)v3 == 17 )
      {
        sub_AE7C90(a1, (__int64)a2);
        return;
      }
      if ( (_BYTE)v3 == 18 )
      {
        sub_AE8440(a1);
        return;
      }
      if ( !(unsigned __int8)sub_AE7F60(a1, (__int64)a2) )
        return;
      v4 = *a2;
      if ( (unsigned int)(v4 - 19) <= 1 || (_BYTE)v4 == 21 )
      {
        v5 = *(a2 - 16);
        if ( (v5 & 2) != 0 )
          v6 = *((_QWORD *)a2 - 4);
        else
          v6 = (__int64)&a2[-8 * ((v5 >> 2) & 0xF) - 16];
        a2 = *(unsigned __int8 **)(v6 + 8);
        if ( !a2 )
          return;
      }
      else
      {
        if ( (_BYTE)v4 != 22 )
          return;
        a2 = (unsigned __int8 *)*((_QWORD *)sub_A17150(a2 - 16) + 1);
        if ( !a2 )
          return;
      }
    }
    sub_AE8230(a1);
  }
}
