// Function: sub_1A22EA0
// Address: 0x1a22ea0
//
char __fastcall sub_1A22EA0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  unsigned int v9; // r15d
  unsigned int v10; // r15d
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  int v13; // r9d
  __int64 v14; // rdi
  unsigned __int8 v15; // r8
  unsigned int v16; // ebx
  int v17; // eax
  unsigned __int64 v18; // rcx
  int v19; // eax
  unsigned int v20; // r14d
  __int64 v21; // rbx
  __int64 v22; // rax
  int v23; // eax
  unsigned int v24; // r14d
  unsigned __int64 v26; // [rsp+8h] [rbp-38h]
  unsigned __int8 v27; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v8 = *(_QWORD *)(a2 + 24 * (2 - v7));
  if ( *(_BYTE *)(v8 + 16) == 13 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 > 0x40 )
    {
      if ( v9 - (unsigned int)sub_16A57B0(v8 + 24) <= 0x40 )
      {
        if ( !**(_QWORD **)(v8 + 24) )
          goto LABEL_8;
        if ( !*(_BYTE *)(a1 + 344) )
          goto LABEL_12;
        goto LABEL_5;
      }
    }
    else if ( !*(_QWORD *)(v8 + 24) )
    {
      goto LABEL_8;
    }
  }
  else
  {
    v8 = 0;
  }
  if ( !*(_BYTE *)(a1 + 344) )
    goto LABEL_12;
LABEL_5:
  v10 = *(_DWORD *)(a1 + 360);
  a3 = *(_QWORD *)(a1 + 368);
  if ( v10 > 0x40 )
  {
    v26 = *(_QWORD *)(a1 + 368);
    v19 = sub_16A57B0(a1 + 352);
    a3 = v26;
    if ( v10 - v19 > 0x40 )
      goto LABEL_8;
    v11 = **(_QWORD **)(a1 + 352);
  }
  else
  {
    v11 = *(_QWORD *)(a1 + 352);
  }
  if ( a3 <= v11 )
  {
LABEL_8:
    LOBYTE(v12) = sub_1A21B40(a1, a2, a3, a4, a5, a6);
    return v12;
  }
LABEL_12:
  v12 = *(_QWORD *)sub_1649C60(*(_QWORD *)(a2 - 24 * v7));
  v14 = *(_QWORD *)(v12 + 24);
  if ( *(_BYTE *)(v14 + 8) == 13 && !byte_4FB3D80 && (LOBYTE(v12) = sub_1A1E0D0(v14, *(_QWORD *)a1), (_BYTE)v12)
    || !*(_BYTE *)(a1 + 344) )
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 8) & 3LL | a2 | 4;
  }
  else
  {
    v15 = v8 != 0;
    if ( v8 )
    {
      v16 = *(_DWORD *)(v8 + 32);
      if ( v16 <= 0x40 )
      {
        v18 = *(_QWORD *)(v8 + 24);
      }
      else
      {
        v17 = sub_16A57B0(v8 + 24);
        v15 = v8 != 0;
        v18 = -1;
        if ( v16 - v17 <= 0x40 )
          v18 = **(_QWORD **)(v8 + 24);
      }
    }
    else
    {
      v20 = *(_DWORD *)(a1 + 360);
      v21 = *(_QWORD *)(a1 + 368);
      if ( v20 > 0x40 )
      {
        v27 = v15;
        v23 = sub_16A57B0(a1 + 352);
        v15 = v27;
        v24 = v20 - v23;
        v22 = -1;
        if ( v24 <= 0x40 )
          v22 = **(_QWORD **)(a1 + 352);
      }
      else
      {
        v22 = *(_QWORD *)(a1 + 352);
      }
      v18 = v21 - v22;
    }
    LOBYTE(v12) = sub_1A22CF0((_QWORD *)a1, a2, a1 + 352, v18, v15, v13);
  }
  return v12;
}
