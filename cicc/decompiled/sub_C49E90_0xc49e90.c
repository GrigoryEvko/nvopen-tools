// Function: sub_C49E90
// Address: 0xc49e90
//
__int64 __fastcall sub_C49E90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r14d
  bool v6; // al
  unsigned int v7; // r15d
  const void *v8; // rax
  unsigned int v10; // eax
  int v11; // eax
  const void *v12; // rax
  unsigned int v13; // eax
  int v14; // eax
  int v17; // ecx
  unsigned __int64 v18; // rax
  int v21; // ecx
  unsigned int v24; // eax
  unsigned int v27; // esi
  unsigned int v28; // eax
  unsigned int v29; // [rsp+Ch] [rbp-34h]

  v5 = *(_DWORD *)(a2 + 8);
  if ( v5 <= 0x40 )
  {
    if ( *(_QWORD *)a2 == *(_QWORD *)a3 )
      goto LABEL_7;
    v6 = *(_QWORD *)a2 == 0;
  }
  else
  {
    if ( sub_C43C50(a2, (const void **)a3) )
      goto LABEL_7;
    v6 = v5 == (unsigned int)sub_C444A0(a2);
  }
  v7 = *(_DWORD *)(a3 + 8);
  if ( v6 )
  {
    v8 = *(const void **)a3;
    *(_DWORD *)(a1 + 8) = v7;
    *(_DWORD *)(a3 + 8) = 0;
    *(_QWORD *)a1 = v8;
    return a1;
  }
  if ( v7 > 0x40 )
  {
    if ( (unsigned int)sub_C444A0(a3) != v7 )
      goto LABEL_11;
LABEL_7:
    *(_DWORD *)(a1 + 8) = v5;
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a2 + 8) = 0;
    return a1;
  }
  if ( !*(_QWORD *)a3 )
    goto LABEL_7;
LABEL_11:
  if ( v5 <= 0x40 )
  {
    _RAX = *(_QWORD *)a2;
    __asm { tzcnt   rcx, rax }
    v24 = 64;
    if ( *(_QWORD *)a2 )
      v24 = _RCX;
    if ( v5 <= v24 )
      v24 = v5;
    v29 = v24;
  }
  else
  {
    v29 = sub_C44590(a2);
  }
  if ( v7 <= 0x40 )
  {
    _RCX = *(const void **)a3;
    v10 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( *(_QWORD *)a3 )
      v10 = _RSI;
    if ( v10 > v7 )
      v10 = v7;
  }
  else
  {
    v10 = sub_C44590(a3);
  }
  if ( v10 < v29 )
  {
    v27 = v29 - v10;
    if ( v5 > 0x40 )
    {
      v29 = v10;
      sub_C482E0(a2, v27);
      v5 = *(_DWORD *)(a2 + 8);
    }
    else if ( v27 == v5 )
    {
      *(_QWORD *)a2 = 0;
      v29 = v10;
    }
    else
    {
      v29 = v10;
      *(_QWORD *)a2 >>= v27;
    }
  }
  else if ( v10 > v29 )
  {
    v28 = v10 - v29;
    if ( v7 > 0x40 )
    {
      sub_C482E0(a3, v28);
      v5 = *(_DWORD *)(a2 + 8);
    }
    else
    {
      if ( v28 == v7 )
        *(_QWORD *)a3 = 0;
      else
        *(_QWORD *)a3 >>= v28;
      v5 = *(_DWORD *)(a2 + 8);
    }
  }
LABEL_17:
  if ( v5 <= 0x40 )
  {
    while ( 1 )
    {
      v12 = *(const void **)a3;
      if ( *(_QWORD *)a2 == *(_QWORD *)a3 )
        break;
LABEL_19:
      if ( (int)sub_C49970(a2, (unsigned __int64 *)a3) <= 0 )
      {
        sub_C46B40(a3, (__int64 *)a2);
        v13 = *(_DWORD *)(a3 + 8);
        if ( v13 <= 0x40 )
        {
          _RDX = *(_QWORD *)a3;
          __asm { tzcnt   rcx, rdx }
          if ( !*(_QWORD *)a3 )
            LODWORD(_RCX) = 64;
          if ( v13 <= (unsigned int)_RCX )
            LODWORD(_RCX) = *(_DWORD *)(a3 + 8);
          v21 = _RCX - v29;
          if ( v13 == v21 )
            *(_QWORD *)a3 = 0;
          else
            *(_QWORD *)a3 = _RDX >> v21;
          v5 = *(_DWORD *)(a2 + 8);
        }
        else
        {
          v14 = sub_C44590(a3);
          sub_C482E0(a3, v14 - v29);
          v5 = *(_DWORD *)(a2 + 8);
        }
        goto LABEL_17;
      }
      sub_C46B40(a2, (__int64 *)a3);
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 <= 0x40 )
      {
        _RAX = *(_QWORD *)a2;
        __asm { tzcnt   rcx, rax }
        if ( !*(_QWORD *)a2 )
          LODWORD(_RCX) = 64;
        if ( v5 <= (unsigned int)_RCX )
          LODWORD(_RCX) = *(_DWORD *)(a2 + 8);
        v17 = _RCX - v29;
        v18 = _RAX >> v17;
        if ( v5 == v17 )
          v18 = 0;
        *(_QWORD *)a2 = v18;
        goto LABEL_17;
      }
      v11 = sub_C44590(a2);
      sub_C482E0(a2, v11 - v29);
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
        goto LABEL_18;
    }
  }
  else
  {
LABEL_18:
    if ( !sub_C43C50(a2, (const void **)a3) )
      goto LABEL_19;
    v12 = *(const void **)a2;
  }
  *(_DWORD *)(a1 + 8) = v5;
  *(_QWORD *)a1 = v12;
  *(_DWORD *)(a2 + 8) = 0;
  return a1;
}
