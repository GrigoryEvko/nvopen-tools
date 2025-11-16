// Function: sub_11AF2C0
// Address: 0x11af2c0
//
bool __fastcall sub_11AF2C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v6; // al
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rcx
  unsigned int v21; // r13d
  unsigned __int64 v22; // rdx

  while ( 1 )
  {
    v6 = *(_BYTE *)a1;
    if ( *(_BYTE *)a2 == 17 )
      break;
    if ( v6 <= 0x15u )
      return sub_AD7630(a1, 0, a3) != 0;
    v8 = 0;
LABEL_5:
    if ( v6 == 91 )
    {
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v10 = *(_QWORD *)(a1 - 8);
      else
        v10 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( **(_BYTE **)(v10 + 64) == 17 )
        return v8 != 0;
      goto LABEL_18;
    }
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v9 )
      return 0;
    v16 = *(_QWORD *)(v9 + 8);
    if ( v6 == 61 && !v16 || v6 == 41 && !v16 )
      return 1;
    if ( !v16 && (unsigned __int8)(v6 - 42) <= 0x11u )
    {
      v17 = *(_QWORD *)(a1 - 64);
      if ( v17 )
      {
        v18 = *(_QWORD *)(a1 - 32);
        if ( v18 )
        {
          LOBYTE(a4) = v16 == 0;
          if ( (unsigned __int8)sub_11AF2C0(v17, a2, v9, a4) || (unsigned __int8)sub_11AF2C0(v18, a2, v19, v20) )
            return 1;
          v9 = *(_QWORD *)(a1 + 16);
        }
      }
    }
LABEL_19:
    if ( !v9 )
      return 0;
    if ( *(_QWORD *)(v9 + 8) )
      return 0;
    if ( (unsigned __int8)(*(_BYTE *)a1 - 82) > 1u )
      return 0;
    v12 = *(_QWORD *)(a1 - 64);
    if ( !v12 )
      return 0;
    v13 = *(_QWORD *)(a1 - 32);
    if ( !v13 )
      return 0;
    sub_B53900(a1);
    if ( (unsigned __int8)sub_11AF2C0(v12, a2, v14, v15) )
      return 1;
    a1 = v13;
  }
  if ( v6 <= 0x15u )
    return 1;
  v8 = a2;
  if ( v6 != 85 )
    goto LABEL_5;
  v11 = *(_QWORD *)(a1 - 32);
  if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(a1 + 80) || *(_DWORD *)(v11 + 36) != 345 )
  {
LABEL_18:
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v9 )
      return 0;
    goto LABEL_19;
  }
  v21 = *(_DWORD *)(a2 + 32);
  if ( v21 > 0x40 )
  {
    if ( v21 - (unsigned int)sub_C444A0(a2 + 24) > 0x40 )
      return 0;
    v22 = **(_QWORD **)(a2 + 24);
  }
  else
  {
    v22 = *(_QWORD *)(a2 + 24);
  }
  return *(unsigned int *)(*(_QWORD *)(a1 + 8) + 32LL) > v22;
}
