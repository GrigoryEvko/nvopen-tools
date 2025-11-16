// Function: sub_A55040
// Address: 0xa55040
//
_BYTE *__fastcall sub_A55040(__int64 a1, _BYTE *a2)
{
  unsigned __int8 *v2; // r13
  size_t v3; // rdx
  size_t v4; // rbx
  _BYTE *v5; // rax
  size_t v6; // rdx
  _BYTE *v7; // rax

  if ( *a2 > 3u )
  {
    v2 = (unsigned __int8 *)sub_BD5D20(a2);
    v4 = v6;
    v7 = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(a1 + 24) )
    {
      sub_CB5D20(a1, 37);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v7 + 1;
      *v7 = 37;
    }
  }
  else
  {
    v2 = (unsigned __int8 *)sub_BD5D20(a2);
    v4 = v3;
    v5 = *(_BYTE **)(a1 + 32);
    if ( (unsigned __int64)v5 >= *(_QWORD *)(a1 + 24) )
    {
      sub_CB5D20(a1, 64);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v5 + 1;
      *v5 = 64;
    }
  }
  return sub_A54F00(a1, v2, v4);
}
