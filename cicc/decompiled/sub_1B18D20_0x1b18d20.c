// Function: sub_1B18D20
// Address: 0x1b18d20
//
bool __fastcall sub_1B18D20(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // r13
  int v8; // eax

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v2 + 16) != 76 )
    return 0;
  v4 = *(_QWORD *)(a2 - 48);
  v5 = *(_QWORD *)(v2 - 48);
  v6 = *(_QWORD *)(a2 - 24);
  v7 = *(_QWORD *)(v2 - 24);
  if ( v5 == v4 && v7 == v6 )
  {
    v8 = *(unsigned __int16 *)(v2 + 18);
  }
  else
  {
    if ( v7 != v4 || v5 != v6 )
      return 0;
    v8 = *(unsigned __int16 *)(v2 + 18);
    if ( v5 != v4 )
    {
      v8 = sub_15FF0F0(v8 & 0xFFFF7FFF);
      goto LABEL_8;
    }
  }
  BYTE1(v8) &= ~0x80u;
LABEL_8:
  result = v5 != 0 && (unsigned int)(v8 - 10) <= 1;
  if ( result )
  {
    **a1 = v5;
    if ( v7 )
    {
      *a1[1] = v7;
      return result;
    }
  }
  return 0;
}
