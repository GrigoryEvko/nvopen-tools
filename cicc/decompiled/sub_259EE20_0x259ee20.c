// Function: sub_259EE20
// Address: 0x259ee20
//
char __fastcall sub_259EE20(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13
  char result; // al
  _BYTE *v4; // rax
  _BYTE *v5; // r14
  unsigned __int64 v6; // rax

  v2 = (_QWORD *)(a1 + 72);
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070((_QWORD *)(a1 + 72)) - 12 <= 1 )
    goto LABEL_2;
  v4 = (_BYTE *)sub_250D070(v2);
  v5 = v4;
  if ( *v4 > 0x1Cu )
  {
    if ( sub_259E650(a1, a2, (unsigned __int64)v4) )
      goto LABEL_3;
    if ( ((*v5 - 62) & 0xFD) == 0 )
    {
      *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96) | *(_BYTE *)(a1 + 97) & 0xFE;
      goto LABEL_3;
    }
LABEL_2:
    *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
    goto LABEL_3;
  }
  if ( !sub_259E650(a1, a2, 0) )
    goto LABEL_2;
LABEL_3:
  if ( (unsigned int)*(unsigned __int8 *)sub_250D070(v2) - 12 > 1 )
  {
    v6 = sub_2509740(v2);
    result = sub_259E650(a1, a2, v6);
    *(_BYTE *)(a1 + 184) = result;
  }
  else
  {
    result = *(_BYTE *)(a1 + 96);
    *(_BYTE *)(a1 + 97) = result;
  }
  return result;
}
