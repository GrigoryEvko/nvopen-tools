// Function: sub_28C4920
// Address: 0x28c4920
//
unsigned __int8 *__fastcall sub_28C4920(__int64 a1, __int64 a2, __int64 **a3)
{
  char v3; // al
  __int64 v5; // rax
  unsigned __int8 *result; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int16 v12; // ax
  int v13; // eax

  v3 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    return 0;
  if ( v3 != 85 )
  {
    if ( v3 != 86 )
      return 0;
    v5 = *(_QWORD *)(a2 - 96);
    if ( *(_BYTE *)v5 != 82 )
      return 0;
    v10 = *(_QWORD *)(a2 - 64);
    v8 = *(_QWORD *)(v5 - 64);
    v11 = *(_QWORD *)(a2 - 32);
    v9 = *(_QWORD *)(v5 - 32);
    if ( v10 == v8 && v11 == v9 )
    {
      v12 = *(_WORD *)(v5 + 2);
    }
    else
    {
      if ( v10 != v9 || v11 != v8 )
        return 0;
      v12 = *(_WORD *)(v5 + 2);
      if ( v10 != v8 )
      {
        v13 = sub_B52870(v12 & 0x3F);
        goto LABEL_19;
      }
    }
    v13 = v12 & 0x3F;
LABEL_19:
    if ( (unsigned int)(v13 - 38) <= 1 && v8 )
      goto LABEL_21;
    return 0;
  }
  v7 = *(_QWORD *)(a2 - 32);
  if ( !v7 )
    return 0;
  if ( *(_BYTE *)v7 )
    return 0;
  if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( (*(_BYTE *)(v7 + 33) & 0x20) == 0 )
    return 0;
  if ( *(_DWORD *)(v7 + 36) != 329 )
    return 0;
  v8 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v9 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v8 )
    return 0;
LABEL_21:
  if ( !v9 )
    return 0;
  *a3 = sub_DD8400(*(_QWORD *)(a1 + 24), a2);
  result = sub_28C2900(a1, a2, v8, v9);
  if ( !result || *result <= 0x1Cu )
  {
    result = sub_28C2900(a1, a2, v9, v8);
    if ( !result || *result <= 0x1Cu )
      return 0;
  }
  return result;
}
