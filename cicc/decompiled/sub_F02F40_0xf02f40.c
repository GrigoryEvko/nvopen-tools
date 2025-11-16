// Function: sub_F02F40
// Address: 0xf02f40
//
__int64 __fastcall sub_F02F40(unsigned __int8 *a1, int a2, __int64 a3, __int64 a4, int a5)
{
  unsigned __int8 *v5; // rax
  unsigned __int8 v6; // dl
  int v8; // eax

  v5 = &a1[a2];
  if ( a2 == 3 )
    goto LABEL_10;
  if ( a2 > 3 )
  {
    if ( a2 != 4 )
      return 0;
    if ( (unsigned __int8)(*(v5 - 1) + 0x80) > 0x3Fu )
      return 0;
    --v5;
LABEL_10:
    if ( (unsigned __int8)(*(v5 - 1) + 0x80) > 0x3Fu )
      return 0;
    v6 = *(v5 - 2);
    a5 = 0;
    if ( (unsigned __int8)(v6 + 0x80) > 0x3Fu )
      return 0;
LABEL_12:
    v8 = *a1;
    if ( (_BYTE)v8 == 0xF0 )
    {
      if ( v6 <= 0x8Fu )
        return 0;
    }
    else if ( (unsigned __int8)v8 > 0xF0u )
    {
      if ( (_BYTE)v8 == 0xF4 && v6 > 0x8Fu )
        return 0;
    }
    else if ( (_BYTE)v8 == 0xE0 )
    {
      if ( v6 <= 0x9Fu )
        return 0;
    }
    else if ( (_BYTE)v8 == 0xED && v6 > 0x9Fu )
    {
      return 0;
    }
    goto LABEL_20;
  }
  if ( a2 != 1 )
  {
    if ( a2 == 2 )
    {
      v6 = *(v5 - 1);
      a5 = 0;
      if ( (unsigned __int8)(v6 + 0x80) > 0x3Fu )
        return 0;
      goto LABEL_12;
    }
    return 0;
  }
  v8 = *a1;
LABEL_20:
  LOBYTE(a5) = (unsigned __int8)(v8 + 0x80) <= 0x41u;
  LOBYTE(v8) = (unsigned __int8)v8 > 0xF4u;
  return (v8 | a5) ^ 1u;
}
