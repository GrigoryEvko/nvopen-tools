// Function: sub_16A7ED0
// Address: 0x16a7ed0
//
__int64 __fastcall sub_16A7ED0(__int64 a1, __int64 a2, __int64 a3, bool *a4)
{
  unsigned int v6; // r13d
  unsigned __int64 v7; // r13
  _QWORD *v8; // rax
  bool v9; // al
  unsigned int v10; // eax
  unsigned int v11; // eax
  int v13; // edx
  unsigned __int64 v14; // r13
  unsigned int v15; // edx
  unsigned int v16; // [rsp+8h] [rbp-38h]
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v6 = *(_DWORD *)(a3 + 8);
  if ( v6 <= 0x40 )
  {
    if ( (unsigned __int64)*(unsigned int *)(a2 + 8) > *(_QWORD *)a3 )
      goto LABEL_3;
LABEL_12:
    *a4 = 1;
    v11 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v11;
    if ( v11 <= 0x40 )
      *(_QWORD *)a1 = 0;
    else
      sub_16A4EF0(a1, 0, 0);
    return a1;
  }
  v17 = *(unsigned int *)(a2 + 8);
  if ( v6 - (unsigned int)sub_16A57B0(a3) > 0x40 || v17 <= **(_QWORD **)a3 )
    goto LABEL_12;
LABEL_3:
  *a4 = 0;
  v7 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v7 <= 0x40 )
  {
    v13 = v7 - 64;
    if ( *(_QWORD *)a2 )
    {
      _BitScanReverse64(&v14, *(_QWORD *)a2);
      v7 = v13 + ((unsigned int)v14 ^ 0x3F);
      v16 = *(_DWORD *)(a3 + 8);
      if ( v16 <= 0x40 )
        goto LABEL_6;
      goto LABEL_19;
    }
  }
  else
  {
    v7 = (unsigned int)sub_16A57B0(a2);
  }
  v16 = *(_DWORD *)(a3 + 8);
  if ( v16 <= 0x40 )
  {
LABEL_6:
    v8 = *(_QWORD **)a3;
    goto LABEL_7;
  }
LABEL_19:
  v15 = v16 - sub_16A57B0(a3);
  v9 = 1;
  if ( v15 > 0x40 )
    goto LABEL_8;
  v8 = **(_QWORD ***)a3;
LABEL_7:
  v9 = v7 < (unsigned __int64)v8;
LABEL_8:
  *a4 = v9;
  v10 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v10;
  if ( v10 > 0x40 )
    sub_16A4FD0(a1, (const void **)a2);
  else
    *(_QWORD *)a1 = *(_QWORD *)a2;
  sub_16A7E20(a1, a3);
  return a1;
}
