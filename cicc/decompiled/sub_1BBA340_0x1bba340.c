// Function: sub_1BBA340
// Address: 0x1bba340
//
__int64 __fastcall sub_1BBA340(__int64 a1, __int64 *a2, int a3)
{
  __int64 v3; // rax
  _BYTE *v4; // r12
  __int64 v6; // r8
  unsigned int v7; // r9d
  __int64 v8; // r13
  int v9; // ecx
  _BYTE *v10; // rsi
  _BYTE *v11; // r10
  unsigned __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // rax
  int v16; // eax
  int v17; // eax
  unsigned __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  unsigned int v20; // [rsp+14h] [rbp-4Ch]
  _BYTE *v21; // [rsp+18h] [rbp-48h]
  int v22; // [rsp+28h] [rbp-38h]
  unsigned int v23; // [rsp+2Ch] [rbp-34h]

  v3 = *a2;
  v4 = *(_BYTE **)(*a2 - 48);
  if ( !a3 )
  {
    v17 = 7;
LABEL_26:
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = v17;
    return a1;
  }
  v6 = (unsigned int)(a3 - 1);
  v7 = *(_QWORD *)(*(_QWORD *)v4 + 32LL);
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = v7;
  while ( 1 )
  {
    v13 = *(_QWORD *)(v3 - 24);
    if ( *(_BYTE *)(v13 + 16) != 13 )
    {
LABEL_8:
      *(_BYTE *)(a1 + 4) = 0;
      return a1;
    }
    v23 = *(_DWORD *)(v13 + 32);
    if ( v23 > 0x40 )
    {
      v18 = v12;
      v19 = v6;
      v20 = v7;
      v22 = v9;
      v21 = v11;
      v16 = sub_16A57B0(v13 + 24);
      v11 = v21;
      v9 = v22;
      v7 = v20;
      v6 = v19;
      v12 = v18;
      if ( v23 - v16 > 0x40 )
        goto LABEL_6;
      v14 = **(_QWORD **)(v13 + 24);
      if ( v18 <= v14 )
        goto LABEL_6;
    }
    else
    {
      v14 = *(_QWORD *)(v13 + 24);
      if ( v12 <= v14 )
        goto LABEL_6;
    }
    if ( v4[16] == 9 )
      goto LABEL_6;
    if ( v10 == v4 || !v10 )
    {
      v10 = v4;
      if ( v9 == 2 )
        goto LABEL_18;
    }
    else
    {
      if ( v11 && v11 != v4 )
        goto LABEL_8;
      v11 = v4;
      if ( v9 == 2 )
      {
LABEL_18:
        v9 = 2;
        goto LABEL_6;
      }
    }
    v9 = 1;
    if ( (_DWORD)v14 != (_DWORD)v8 )
      v9 = 2;
LABEL_6:
    if ( v6 == v8 )
      break;
    v3 = a2[++v8];
    v4 = *(_BYTE **)(v3 - 48);
    if ( v7 != *(_DWORD *)(*(_QWORD *)v4 + 32LL) )
      goto LABEL_8;
  }
  if ( v9 != 1 || !v11 )
  {
    v17 = (v11 == 0) + 6;
    goto LABEL_26;
  }
  *(_BYTE *)(a1 + 4) = 1;
  *(_DWORD *)a1 = 2;
  return a1;
}
