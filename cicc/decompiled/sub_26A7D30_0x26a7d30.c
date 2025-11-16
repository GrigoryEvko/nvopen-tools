// Function: sub_26A7D30
// Address: 0x26a7d30
//
__int64 __fastcall sub_26A7D30(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int8 v10; // al
  __int64 v11; // rax
  int v12; // ebx
  __int64 *v13; // r14
  int v14; // eax
  int v15; // edx
  bool v16; // zf
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+8h] [rbp-58h]
  char v23; // [rsp+17h] [rbp-49h]
  __int64 *v24; // [rsp+18h] [rbp-48h]

  v22 = *(_QWORD *)(a1 + 104);
  v23 = *(_BYTE *)(a1 + 112);
  v7 = *(_QWORD *)(a1 + 72);
  v8 = v7 & 3;
  v9 = v7 & 0xFFFFFFFFFFFFFFFCLL;
  if ( v8 == 3 )
    v9 = *(_QWORD *)(v9 + 24);
  v10 = *(_BYTE *)v9;
  if ( *(_BYTE *)v9 )
  {
    if ( v10 == 22 )
    {
      v9 = *(_QWORD *)(v9 + 24);
    }
    else if ( v10 <= 0x1Cu )
    {
      v9 = 0;
    }
    else
    {
      v9 = sub_B43CB0(v9);
    }
  }
  nullsub_1518();
  v11 = sub_26A73D0(a2, v9 & 0xFFFFFFFFFFFFFFFCLL, 0, a1, 0, 1);
  if ( v11 && *(_BYTE *)(v11 + 337) )
  {
    v12 = -1;
    v13 = *(__int64 **)(v11 + 376);
    v24 = &v13[*(unsigned int *)(v11 + 384)];
    if ( v24 == v13 )
    {
      if ( *(_BYTE *)(a1 + 112) == v23 )
      {
        if ( !v23 )
          return 1;
        return *(_QWORD *)(a1 + 104) == v22;
      }
    }
    else
    {
      do
      {
        v14 = sub_B2D810(*v13, a3, a4, -1);
        v15 = v12;
        v12 = v14;
        if ( v14 == -1 || v15 != -1 && v15 != v14 )
        {
          v16 = *(_BYTE *)(a1 + 112) == 0;
          *(_QWORD *)(a1 + 104) = 0;
          if ( v16 )
            *(_BYTE *)(a1 + 112) = 1;
          goto LABEL_19;
        }
        ++v13;
      }
      while ( v24 != v13 );
      v18 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
        v18 = *(_QWORD *)(v18 + 24);
      v19 = (_QWORD *)sub_BD5C60(v18);
      v20 = sub_BCB2D0(v19);
      v21 = sub_ACD640(v20, v12, 0);
      v16 = *(_BYTE *)(a1 + 112) == 0;
      *(_QWORD *)(a1 + 104) = v21;
      if ( v16 )
        *(_BYTE *)(a1 + 112) = 1;
      if ( v23 )
        return *(_QWORD *)(a1 + 104) == v22;
    }
    return 0;
  }
  if ( *(_BYTE *)(a1 + 112) )
  {
    *(_QWORD *)(a1 + 104) = 0;
  }
  else
  {
    *(_QWORD *)(a1 + 104) = 0;
    *(_BYTE *)(a1 + 112) = 1;
  }
LABEL_19:
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}
