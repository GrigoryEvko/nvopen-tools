// Function: sub_19C0FD0
// Address: 0x19c0fd0
//
bool __fastcall sub_19C0FD0(__int64 a1)
{
  __int64 v1; // r8
  bool result; // al
  char v3; // r9
  __int64 v4; // rdi
  char v5; // r10
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rax

  v1 = *(_QWORD *)(a1 - 48);
  result = 1;
  v3 = *(_BYTE *)(v1 + 16);
  if ( v3 == 9 )
    return result;
  v4 = *(_QWORD *)(a1 - 24);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 == 9 )
    return result;
  if ( v3 == 77 )
  {
    v6 = 0;
    if ( v5 == 77 )
      v6 = v4;
    v7 = 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    {
      v8 = *(_QWORD *)(v1 - 8);
      v9 = v8 + v7;
    }
    else
    {
      v9 = v1;
      v8 = v1 - v7;
    }
    if ( v8 != v9 )
    {
      while ( *(_BYTE *)(*(_QWORD *)v8 + 16LL) != 9 )
      {
        v8 += 24;
        if ( v9 == v8 )
          goto LABEL_21;
      }
      return 1;
    }
LABEL_21:
    if ( !v6 )
    {
LABEL_19:
      if ( v5 != 79 )
        return 0;
      goto LABEL_33;
    }
    goto LABEL_22;
  }
  if ( v5 == 77 )
  {
    v6 = v4;
LABEL_22:
    v10 = 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
    {
      v12 = *(_QWORD *)(v6 - 8);
      v11 = v12 + v10;
    }
    else
    {
      v11 = v6;
      v12 = v6 - v10;
    }
    while ( v11 != v12 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v12 + 16LL) == 9 )
        return 1;
      v12 += 24;
    }
  }
  if ( v3 != 79 )
    goto LABEL_19;
  if ( v5 != 79 )
    v4 = 0;
  result = 1;
  if ( *(_BYTE *)(*(_QWORD *)(v1 - 48) + 16LL) != 9 && *(_BYTE *)(*(_QWORD *)(v1 - 24) + 16LL) != 9 )
  {
    if ( !v4 )
      return 0;
LABEL_33:
    result = 1;
    if ( *(_BYTE *)(*(_QWORD *)(v4 - 48) + 16LL) != 9 )
      return *(_BYTE *)(*(_QWORD *)(v4 - 24) + 16LL) == 9;
  }
  return result;
}
