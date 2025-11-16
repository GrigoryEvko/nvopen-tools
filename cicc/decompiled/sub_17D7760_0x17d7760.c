// Function: sub_17D7760
// Address: 0x17d7760
//
__int64 __fastcall sub_17D7760(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 *v7; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int128 v10; // rdi
  __int64 v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  __int64 v22; // rdi
  const char *v23; // rax
  size_t v24; // rdx
  _BYTE *v25; // rdi
  char *v26; // rsi
  _BYTE *v27; // rax
  size_t v28; // r15
  _QWORD *v29; // rax
  _DWORD *v30; // rdx
  __int64 v31; // r14
  _BYTE *v32; // rax
  _QWORD *v33; // rax
  _DWORD *v34; // rdx
  char *v35; // rax
  char *v36; // r15
  size_t v37; // rax
  size_t v38; // rbx
  _BYTE *v39; // rbx

  if ( !byte_4FA4520 )
    goto LABEL_2;
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v19 = sub_16E8CB0();
    v20 = v19[3];
    v21 = (__int64)v19;
    if ( (unsigned __int64)(v19[2] - v20) <= 8 )
    {
      v21 = sub_16E7EE0((__int64)v19, "ZZZ call ", 9u);
    }
    else
    {
      *(_BYTE *)(v20 + 8) = 32;
      *(_QWORD *)v20 = 0x6C6C6163205A5A5ALL;
      v19[3] += 9LL;
    }
    v22 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v22 + 16) )
      v22 = 0;
    v23 = sub_1649960(v22);
    v25 = *(_BYTE **)(v21 + 24);
    v26 = (char *)v23;
    v27 = *(_BYTE **)(v21 + 16);
    v28 = v24;
    if ( v24 <= v27 - v25 )
    {
      if ( v24 )
      {
        memcpy(v25, v26, v24);
        v27 = *(_BYTE **)(v21 + 16);
        v25 = (_BYTE *)(v28 + *(_QWORD *)(v21 + 24));
        *(_QWORD *)(v21 + 24) = v25;
      }
      goto LABEL_25;
    }
    goto LABEL_37;
  }
  v33 = sub_16E8CB0();
  v34 = (_DWORD *)v33[3];
  v21 = (__int64)v33;
  if ( v33[2] - (_QWORD)v34 <= 3u )
  {
    v21 = sub_16E7EE0((__int64)v33, "ZZZ ", 4u);
  }
  else
  {
    *v34 = 542792282;
    v33[3] += 4LL;
  }
  v35 = sub_15F29F0((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24);
  v36 = v35;
  if ( !v35 )
  {
LABEL_38:
    v25 = *(_BYTE **)(v21 + 24);
    if ( *(_BYTE **)(v21 + 16) != v25 )
      goto LABEL_26;
LABEL_39:
    sub_16E7EE0(v21, "\n", 1u);
    goto LABEL_27;
  }
  v37 = strlen(v35);
  v25 = *(_BYTE **)(v21 + 24);
  v38 = v37;
  v27 = *(_BYTE **)(v21 + 16);
  if ( v38 > v27 - v25 )
  {
    v24 = v38;
    v26 = v36;
LABEL_37:
    v21 = sub_16E7EE0(v21, v26, v24);
    goto LABEL_38;
  }
  if ( v38 )
  {
    memcpy(v25, v36, v38);
    v39 = (_BYTE *)(*(_QWORD *)(v21 + 24) + v38);
    v27 = *(_BYTE **)(v21 + 16);
    *(_QWORD *)(v21 + 24) = v39;
    v25 = v39;
  }
LABEL_25:
  if ( v27 == v25 )
    goto LABEL_39;
LABEL_26:
  *v25 = 10;
  ++*(_QWORD *)(v21 + 24);
LABEL_27:
  v29 = sub_16E8CB0();
  v30 = (_DWORD *)v29[3];
  v31 = (__int64)v29;
  if ( v29[2] - (_QWORD)v30 <= 3u )
  {
    v31 = sub_16E7EE0((__int64)v29, "QQQ ", 4u);
  }
  else
  {
    *v30 = 542200145;
    v29[3] += 4LL;
  }
  sub_155C2B0(a2, v31, 0);
  v32 = *(_BYTE **)(v31 + 24);
  if ( *(_BYTE **)(v31 + 16) == v32 )
  {
    sub_16E7EE0(v31, "\n", 1u);
  }
  else
  {
    *v32 = 10;
    ++*(_QWORD *)(v31 + 24);
  }
LABEL_2:
  v4 = 0;
  v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  while ( v5 != v4 )
  {
    while ( 1 )
    {
      v6 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v7 = *(__int64 **)(v6 + 24 * v4);
      v8 = *(unsigned __int8 *)(*v7 + 8);
      if ( (unsigned __int8)v8 <= 0xFu )
      {
        v9 = 35454;
        if ( _bittest64(&v9, v8) )
          break;
      }
      if ( ((unsigned int)(v8 - 13) <= 1 || (_DWORD)v8 == 16) && sub_16435F0(*v7, 0) )
        break;
      if ( v5 == ++v4 )
        goto LABEL_14;
    }
    *((_QWORD *)&v10 + 1) = v7;
    *(_QWORD *)&v10 = a1;
    ++v4;
    sub_17D5820(v10, a2);
  }
LABEL_14:
  v11 = *(_QWORD *)a2;
  v12 = sub_17CD8D0(a1, *(_QWORD *)a2);
  v14 = (__int64)v12;
  if ( v12 )
    v14 = sub_15A06D0((__int64 **)v12, v11, (__int64)v12, v13);
  sub_17D4920((__int64)a1, (__int64 *)a2, v14);
  v17 = sub_15A06D0(*(__int64 ***)(a1[1] + 184LL), a2, v15, v16);
  return sub_17D4B80((__int64)a1, a2, v17);
}
