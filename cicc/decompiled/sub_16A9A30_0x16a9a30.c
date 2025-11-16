// Function: sub_16A9A30
// Address: 0x16a9a30
//
__int64 __fastcall sub_16A9A30(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r14d
  bool v6; // al
  unsigned int v7; // r15d
  const void *v8; // rax
  unsigned int v12; // eax
  unsigned int v14; // eax
  int v18; // ecx
  unsigned __int64 v19; // rax
  const void *v20; // rax
  unsigned int v21; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // eax
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
    if ( sub_16A5220(a2, (const void **)a3) )
      goto LABEL_7;
    v6 = v5 == (unsigned int)sub_16A57B0(a2);
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
    if ( (unsigned int)sub_16A57B0(a3) != v7 )
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
  if ( v5 > 0x40 )
  {
    v29 = sub_16A58A0(a2);
  }
  else
  {
    _RAX = *(_QWORD *)a2;
    __asm { tzcnt   rcx, rax }
    v12 = 64;
    if ( *(_QWORD *)a2 )
      v12 = _RCX;
    if ( v5 <= v12 )
      v12 = v5;
    v29 = v12;
  }
  if ( v7 > 0x40 )
  {
    v14 = sub_16A58A0(a3);
  }
  else
  {
    _RCX = *(const void **)a3;
    v14 = 64;
    __asm { tzcnt   rsi, rcx }
    if ( *(_QWORD *)a3 )
      v14 = _RSI;
    if ( v14 > v7 )
      v14 = v7;
  }
  if ( v14 < v29 )
  {
    v27 = v29 - v14;
    if ( v5 > 0x40 )
    {
      v29 = v14;
      sub_16A8110(a2, v27);
      v5 = *(_DWORD *)(a2 + 8);
    }
    else if ( v27 == v5 )
    {
      *(_QWORD *)a2 = 0;
      v29 = v14;
    }
    else
    {
      v29 = v14;
      *(_QWORD *)a2 >>= v27;
    }
  }
  else if ( v14 > v29 )
  {
    v28 = v14 - v29;
    if ( v7 > 0x40 )
    {
      sub_16A8110(a3, v28);
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
LABEL_24:
  if ( v5 <= 0x40 )
  {
    while ( 1 )
    {
      v20 = *(const void **)a3;
      if ( *(_QWORD *)a2 == *(_QWORD *)a3 )
        break;
LABEL_26:
      if ( (int)sub_16A9900(a2, (unsigned __int64 *)a3) <= 0 )
      {
        sub_16A7590(a3, (__int64 *)a2);
        v21 = *(_DWORD *)(a3 + 8);
        if ( v21 > 0x40 )
        {
          v26 = sub_16A58A0(a3);
          sub_16A8110(a3, v26 - v29);
          v5 = *(_DWORD *)(a2 + 8);
        }
        else
        {
          _RDX = *(_QWORD *)a3;
          __asm { tzcnt   rcx, rdx }
          if ( !*(_QWORD *)a3 )
            LODWORD(_RCX) = 64;
          if ( (unsigned int)_RCX > v21 )
            LODWORD(_RCX) = *(_DWORD *)(a3 + 8);
          v24 = _RCX - v29;
          if ( v21 == v24 )
            *(_QWORD *)a3 = 0;
          else
            *(_QWORD *)a3 = _RDX >> v24;
          v5 = *(_DWORD *)(a2 + 8);
        }
        goto LABEL_24;
      }
      sub_16A7590(a2, (__int64 *)a3);
      v5 = *(_DWORD *)(a2 + 8);
      if ( v5 > 0x40 )
      {
        v25 = sub_16A58A0(a2);
        sub_16A8110(a2, v25 - v29);
        v5 = *(_DWORD *)(a2 + 8);
        goto LABEL_24;
      }
      _RAX = *(_QWORD *)a2;
      __asm { tzcnt   rcx, rax }
      if ( !*(_QWORD *)a2 )
        LODWORD(_RCX) = 64;
      if ( v5 <= (unsigned int)_RCX )
        LODWORD(_RCX) = *(_DWORD *)(a2 + 8);
      v18 = _RCX - v29;
      v19 = _RAX >> v18;
      if ( v5 == v18 )
        v19 = 0;
      *(_QWORD *)a2 = v19;
    }
  }
  else
  {
    if ( !sub_16A5220(a2, (const void **)a3) )
      goto LABEL_26;
    v20 = *(const void **)a2;
  }
  *(_DWORD *)(a1 + 8) = v5;
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a2 + 8) = 0;
  return a1;
}
