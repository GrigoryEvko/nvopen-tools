// Function: sub_BD3080
// Address: 0xbd3080
//
__int64 __fastcall sub_BD3080(unsigned __int8 *a1, _QWORD *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 v5; // rax
  __int64 v7; // rax

  *a2 = 0;
  v2 = *a1;
  if ( *a1 <= 0x1Cu )
  {
    if ( v2 == 23 )
    {
      v5 = *((_QWORD *)a1 + 9);
      v4 = 0;
      if ( !v5 )
        return v4;
      goto LABEL_4;
    }
    if ( v2 > 3u )
    {
      v4 = 1;
      if ( v2 == 22 )
      {
        v5 = *((_QWORD *)a1 + 3);
        v4 = 0;
        if ( v5 )
          goto LABEL_4;
      }
    }
    else
    {
      v7 = *((_QWORD *)a1 + 5);
      v4 = 0;
      if ( v7 )
        *a2 = *(_QWORD *)(v7 + 120);
    }
  }
  else
  {
    v3 = *((_QWORD *)a1 + 5);
    v4 = 0;
    if ( v3 )
    {
      v5 = *(_QWORD *)(v3 + 72);
      if ( v5 )
LABEL_4:
        *a2 = *(_QWORD *)(v5 + 112);
    }
  }
  return v4;
}
