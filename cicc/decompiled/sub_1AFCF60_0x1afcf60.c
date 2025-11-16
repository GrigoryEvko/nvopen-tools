// Function: sub_1AFCF60
// Address: 0x1afcf60
//
__int64 __fastcall sub_1AFCF60(__int64 a1, unsigned int a2)
{
  __int64 v3; // r11
  char v4; // di
  unsigned int v5; // edx
  unsigned int v6; // eax
  int v7; // r8d
  __int64 result; // rax
  unsigned int v9; // eax
  unsigned int v10; // r8d
  unsigned int v11; // edx
  int v12; // r8d
  int v13; // r10d
  int v14; // esi
  int v15; // r10d
  int v16; // esi
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 *v22; // rdi

  v3 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 != 19 )
  {
    result = a1;
    if ( a2 <= 1 )
      return result;
    v5 = 0;
    v9 = 0;
    goto LABEL_35;
  }
  v5 = *(_DWORD *)(v3 + 24);
  v6 = v5 >> 1;
  if ( (v5 & 1) == 0 )
  {
    if ( (v5 & 0x40) != 0 )
      v6 = v5 >> 14;
    else
      v6 = v5 >> 7;
  }
  if ( v6 && (v6 & 1) == 0 )
  {
    v7 = (v6 >> 1) & 0x1F;
    if ( ((v6 >> 1) & 0x20) == 0 )
    {
      a2 *= v7;
      result = a1;
      if ( a2 <= 1 )
        return result;
      goto LABEL_8;
    }
    a2 *= v7 | (v6 >> 2) & 0xFE0;
  }
  result = a1;
  if ( a2 <= 1 )
    return result;
LABEL_8:
  v9 = 0;
  v10 = v5 >> 1;
  if ( (*(_DWORD *)(v3 + 24) & 1) != 0 )
    goto LABEL_9;
  v9 = (v5 >> 1) & 0x1F;
  if ( (v10 & 0x20) != 0 )
    v9 |= (v5 >> 2) & 0xFE0;
LABEL_35:
  v10 = v5 >> ((v5 & 0x40) == 0 ? 7 : 14);
LABEL_9:
  v11 = v10 >> 1;
  if ( (v10 & 1) == 0 )
    v11 = v10 >> ((v10 & 0x40) == 0 ? 7 : 14);
  v12 = v11 & 0x1F;
  if ( (v11 & 0x20) != 0 )
    v12 |= (v11 >> 1) & 0xFE0;
  v13 = v12 << 7;
  if ( a2 > 0x1F )
  {
    v13 = v12 << 14;
    if ( (a2 & 0xFE0) != 0 )
      a2 = (2 * (a2 & 0xFFF)) & 0x1FC0 | a2 & 0x1F | 0x20;
    else
      a2 &= 0xFFFu;
  }
  v14 = v13 | (2 * a2);
  v15 = 2 * v14 + 1;
  if ( v9 )
  {
    if ( v9 > 0x1F )
    {
      v16 = v14 << 14;
      v9 = (2 * (_WORD)v9) & 0x1FC0 | v9 & 0x1F | 0x20;
    }
    else
    {
      v16 = v14 << 7;
    }
    v15 = v16 | (2 * v9);
  }
  v17 = v3;
  if ( v4 == 19 )
  {
    v18 = v3;
    do
    {
      if ( !*(_DWORD *)(v18 + 24) )
        break;
      v18 = *(_QWORD *)(v18 + 8 * (1LL - *(unsigned int *)(v18 + 8)));
    }
    while ( *(_BYTE *)v18 == 19 );
  }
  else
  {
    if ( v4 == 15 )
      goto LABEL_25;
    v18 = v3;
  }
  v17 = *(_QWORD *)(v3 - 8LL * *(unsigned int *)(v3 + 8));
  v3 = v18;
LABEL_25:
  v19 = (__int64 *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
    v19 = (__int64 *)*v19;
  v20 = sub_15C0C90(v19, v3, v17, v15, 0, 1);
  v21 = 0;
  if ( *(_DWORD *)(a1 + 8) == 2 )
    v21 = *(_QWORD *)(a1 - 8);
  v22 = (__int64 *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
    v22 = (__int64 *)*v22;
  return sub_15B9E00(v22, *(_DWORD *)(a1 + 4), *(unsigned __int16 *)(a1 + 2), v20, v21, 0, 1);
}
