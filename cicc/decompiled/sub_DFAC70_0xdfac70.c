// Function: sub_DFAC70
// Address: 0xdfac70
//
char __fastcall sub_DFAC70(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rax
  char result; // al
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rcx
  _BYTE *v10; // rdi

  v2 = *(__int64 (**)(void))(**(_QWORD **)a1 + 856LL);
  if ( (char *)v2 != (char *)sub_DF6E40 )
    return v2();
  if ( *(_BYTE *)a2 != 86 )
    return 0;
  v4 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( **(_BYTE **)(v4 + 32) <= 0x15u && **(_BYTE **)(v4 + 64) <= 0x15u )
    return 0;
  v5 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  if ( !sub_BCAC40(v5, 1) )
    goto LABEL_27;
  if ( *(_BYTE *)a2 == 57 )
    return 0;
  v6 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)a2 == 86 && *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) == v6 && **(_BYTE **)(a2 - 32) <= 0x15u )
  {
    if ( !sub_AC30F0(*(_QWORD *)(a2 - 32)) )
    {
LABEL_27:
      v6 = *(_QWORD *)(a2 + 8);
      goto LABEL_15;
    }
    return 0;
  }
LABEL_15:
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  result = sub_BCAC40(v6, 1);
  if ( !result )
    return 1;
  if ( *(_BYTE *)a2 == 58 )
    return 0;
  if ( *(_BYTE *)a2 == 86 )
  {
    v8 = *(_QWORD *)(a2 - 96);
    v9 = *(_QWORD *)(a2 + 8);
    if ( *(_QWORD *)(v8 + 8) == v9 )
    {
      v10 = *(_BYTE **)(a2 - 64);
      if ( *v10 <= 0x15u )
        return !sub_AD7A80(v10, 1, v8, v9, v7);
    }
  }
  return result;
}
