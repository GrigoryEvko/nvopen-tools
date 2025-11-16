// Function: sub_154B790
// Address: 0x154b790
//
_BYTE *__fastcall sub_154B790(__int64 a1, __int64 a2)
{
  const char *v2; // r13
  size_t v3; // rdx
  size_t v4; // rbx
  _BYTE *v5; // rax
  size_t v6; // rdx
  _BYTE *v7; // rax

  if ( *(_BYTE *)(a2 + 16) > 3u )
  {
    v2 = (const char *)sub_1649960(a2);
    v4 = v6;
    v7 = *(_BYTE **)(a1 + 24);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(a1 + 16) )
    {
      sub_16E7DE0(a1, 37);
    }
    else
    {
      *(_QWORD *)(a1 + 24) = v7 + 1;
      *v7 = 37;
    }
  }
  else
  {
    v2 = (const char *)sub_1649960(a2);
    v4 = v3;
    v5 = *(_BYTE **)(a1 + 24);
    if ( (unsigned __int64)v5 >= *(_QWORD *)(a1 + 16) )
    {
      sub_16E7DE0(a1, 64);
    }
    else
    {
      *(_QWORD *)(a1 + 24) = v5 + 1;
      *v5 = 64;
    }
  }
  return sub_154B650(a1, v2, v4);
}
