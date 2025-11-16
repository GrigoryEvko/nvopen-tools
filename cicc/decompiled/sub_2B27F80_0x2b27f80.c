// Function: sub_2B27F80
// Address: 0x2b27f80
//
bool __fastcall sub_2B27F80(_QWORD **a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdi
  _QWORD *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rcx
  _BYTE *v10; // rdi

  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( !sub_BCAC40(v4, 1) )
    goto LABEL_15;
  if ( *(_BYTE *)a2 != 57 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    if ( *(_BYTE *)a2 != 86 || *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) != v5 || **(_BYTE **)(a2 - 32) > 0x15u )
    {
LABEL_16:
      if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
        v5 = **(_QWORD **)(v5 + 16);
      if ( sub_BCAC40(v5, 1) )
      {
        if ( *(_BYTE *)a2 == 58 )
          goto LABEL_11;
        if ( *(_BYTE *)a2 == 86 )
        {
          v9 = *(_QWORD *)(a2 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) == v9 )
          {
            v10 = *(_BYTE **)(a2 - 64);
            if ( *v10 <= 0x15u )
            {
              if ( !sub_AD7A80(v10, 1, v7, v9, v8) )
                return 0;
              goto LABEL_11;
            }
          }
        }
      }
      return 0;
    }
    if ( sub_AC30F0(*(_QWORD *)(a2 - 32)) )
      goto LABEL_11;
LABEL_15:
    v5 = *(_QWORD *)(a2 + 8);
    goto LABEL_16;
  }
LABEL_11:
  sub_2B27770(a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v6 = *(_QWORD **)(a2 - 8);
  else
    v6 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  return **a1 == *v6;
}
