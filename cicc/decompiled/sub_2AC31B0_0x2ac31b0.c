// Function: sub_2AC31B0
// Address: 0x2ac31b0
//
void __fastcall sub_2AC31B0(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rdx
  _QWORD *v3; // rax
  _QWORD *i; // rdx
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *j; // rdx
  unsigned int v9; // ecx
  unsigned int v10; // eax
  int v11; // r13d
  unsigned int v12; // eax
  unsigned int v13; // ecx
  unsigned int v14; // eax
  int v15; // r13d
  unsigned int v16; // eax

  v1 = *(_DWORD *)(a1 + 368);
  ++*(_QWORD *)(a1 + 352);
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 372) )
      goto LABEL_7;
    v2 = *(unsigned int *)(a1 + 376);
    if ( (unsigned int)v2 <= 0x40 )
      goto LABEL_4;
    sub_C7D6A0(*(_QWORD *)(a1 + 360), 40 * v2, 8);
    *(_DWORD *)(a1 + 376) = 0;
LABEL_35:
    *(_QWORD *)(a1 + 360) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 368) = 0;
    goto LABEL_7;
  }
  v13 = 4 * v1;
  v2 = *(unsigned int *)(a1 + 376);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v13 = 64;
  if ( (unsigned int)v2 <= v13 )
  {
LABEL_4:
    v3 = *(_QWORD **)(a1 + 360);
    for ( i = &v3[5 * v2]; i != v3; *((_BYTE *)v3 - 28) = 1 )
    {
      *v3 = -4096;
      v3 += 5;
      *((_DWORD *)v3 - 8) = -1;
    }
    goto LABEL_6;
  }
  v14 = v1 - 1;
  if ( v14 )
  {
    _BitScanReverse(&v14, v14);
    v15 = 1 << (33 - (v14 ^ 0x1F));
    if ( v15 < 64 )
      v15 = 64;
    if ( v15 == (_DWORD)v2 )
      goto LABEL_33;
  }
  else
  {
    v15 = 64;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 360), 40 * v2, 8);
  v16 = sub_2AAAC60(v15);
  *(_DWORD *)(a1 + 376) = v16;
  if ( !v16 )
    goto LABEL_35;
  *(_QWORD *)(a1 + 360) = sub_C7D670(40LL * v16, 8);
LABEL_33:
  sub_2AC3110(a1 + 352);
LABEL_7:
  v5 = *(_DWORD *)(a1 + 400);
  ++*(_QWORD *)(a1 + 384);
  if ( v5 )
  {
    v9 = 4 * v5;
    v6 = *(unsigned int *)(a1 + 408);
    if ( (unsigned int)(4 * v5) < 0x40 )
      v9 = 64;
    if ( (unsigned int)v6 <= v9 )
      goto LABEL_10;
    v10 = v5 - 1;
    if ( v10 )
    {
      _BitScanReverse(&v10, v10);
      v11 = 1 << (33 - (v10 ^ 0x1F));
      if ( v11 < 64 )
        v11 = 64;
      if ( v11 == (_DWORD)v6 )
        goto LABEL_23;
    }
    else
    {
      v11 = 64;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 392), (unsigned __int64)(unsigned int)v6 << 6, 8);
    v12 = sub_2AAAC60(v11);
    *(_DWORD *)(a1 + 408) = v12;
    if ( !v12 )
      goto LABEL_37;
    *(_QWORD *)(a1 + 392) = sub_C7D670((unsigned __int64)v12 << 6, 8);
LABEL_23:
    sub_2AC3160(a1 + 384);
    goto LABEL_13;
  }
  if ( *(_DWORD *)(a1 + 404) )
  {
    v6 = *(unsigned int *)(a1 + 408);
    if ( (unsigned int)v6 <= 0x40 )
    {
LABEL_10:
      v7 = *(_QWORD **)(a1 + 392);
      for ( j = &v7[8 * v6]; j != v7; *((_BYTE *)v7 - 52) = 1 )
      {
        *v7 = -4096;
        v7 += 8;
        *((_DWORD *)v7 - 14) = -1;
      }
      goto LABEL_12;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 392), (unsigned __int64)(unsigned int)v6 << 6, 8);
    *(_DWORD *)(a1 + 408) = 0;
LABEL_37:
    *(_QWORD *)(a1 + 392) = 0;
LABEL_12:
    *(_QWORD *)(a1 + 400) = 0;
  }
LABEL_13:
  sub_2AC2F10(a1 + 160);
  sub_2AC2F10(a1 + 192);
}
