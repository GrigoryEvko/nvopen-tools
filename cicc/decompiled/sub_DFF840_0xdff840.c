// Function: sub_DFF840
// Address: 0xdff840
//
__int64 __fastcall sub_DFF840(__int64 a1, _BYTE *a2)
{
  bool v3; // r12
  int v4; // eax
  unsigned int v5; // ecx
  unsigned int v6; // ebx
  int v7; // r15d
  _BYTE **v8; // rax
  _BYTE *v9; // rdi
  unsigned int v11; // [rsp+8h] [rbp-38h]
  unsigned int v12; // [rsp+Ch] [rbp-34h]

  v3 = (*(_BYTE *)(a1 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a1 - 16) & 2) == 0 )
  {
    v12 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
    if ( v12 <= 2 )
      goto LABEL_3;
    if ( (unsigned __int8)(**(_BYTE **)(a1 - 8LL * ((*(_BYTE *)(a1 - 16) >> 2) & 0xF) - 16) - 5) > 0x1Fu )
      goto LABEL_17;
LABEL_20:
    v4 = 3;
    v5 = 3;
LABEL_4:
    v11 = (v12 - v4) / v5;
    if ( v12 - v4 < v5 )
      return 0;
    goto LABEL_5;
  }
  v12 = *(_DWORD *)(a1 - 24);
  if ( v12 <= 2 )
  {
LABEL_3:
    v4 = 1;
    v5 = 2;
    goto LABEL_4;
  }
  if ( (unsigned __int8)(***(_BYTE ***)(a1 - 32) - 5) <= 0x1Fu )
    goto LABEL_20;
LABEL_17:
  v11 = (v12 - 1) >> 1;
LABEL_5:
  v6 = 1;
  v7 = 0;
  while ( 1 )
  {
    if ( v3 )
    {
      v8 = *(_BYTE ***)(a1 - 32);
      if ( v12 <= 2 )
        goto LABEL_8;
    }
    else
    {
      v8 = (_BYTE **)(a1 + -16 - 8LL * ((*(_BYTE *)(a1 - 16) >> 2) & 0xF));
      if ( v12 <= 2 )
        goto LABEL_8;
    }
    if ( (unsigned __int8)(**v8 - 5) <= 0x1Fu )
      break;
LABEL_8:
    v9 = v8[v6];
    if ( v9 == a2 )
      return 1;
LABEL_9:
    if ( (unsigned __int8)sub_DFF840(v9, a2) )
      return 1;
    ++v7;
    v6 += 2;
    if ( v7 == v11 )
      return 0;
  }
  v9 = v8[3 * v7 + 3];
  if ( v9 != a2 )
    goto LABEL_9;
  return 1;
}
