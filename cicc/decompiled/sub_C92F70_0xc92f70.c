// Function: sub_C92F70
// Address: 0xc92f70
//
__int64 __fastcall sub_C92F70(__int64 *a1)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx

  if ( !a1[1] )
    return 10;
  if ( (unsigned __int8)sub_C92F10(a1, (__int64)"0x", 2u) )
  {
    v2 = a1[1];
    v3 = *a1;
    v4 = 0;
    if ( v2 > 1 )
    {
      v4 = v2 - 2;
      v2 = 2;
    }
    a1[1] = v4;
    *a1 = v3 + v2;
    return 16;
  }
  else if ( (unsigned __int8)sub_C92F10(a1, (__int64)"0b", 2u) )
  {
    v5 = a1[1];
    v6 = *a1;
    v7 = 0;
    if ( v5 > 1 )
    {
      v7 = v5 - 2;
      v5 = 2;
    }
    a1[1] = v7;
    *a1 = v6 + v5;
    return 2;
  }
  else
  {
    v8 = a1[1];
    v9 = *a1;
    if ( v8 > 1 )
    {
      if ( *(_WORD *)v9 == 28464 )
      {
        a1[1] = v8 - 2;
        *a1 = v9 + 2;
        return 8;
      }
      if ( *(_BYTE *)v9 == 48 && (unsigned __int8)(*(_BYTE *)(v9 + 1) - 48) <= 9u )
      {
        a1[1] = v8 - 1;
        *a1 = v9 + 1;
        return 8;
      }
    }
    return 10;
  }
}
