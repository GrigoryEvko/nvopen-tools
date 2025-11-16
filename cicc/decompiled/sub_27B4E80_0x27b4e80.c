// Function: sub_27B4E80
// Address: 0x27b4e80
//
unsigned __int64 __fastcall sub_27B4E80(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned int v8; // ecx
  _QWORD **v9; // rax
  _QWORD *v10; // rdi
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // r15
  const void *v14; // r8
  __int64 v15; // rdx
  unsigned __int64 v16; // rsi
  void *v17; // rcx
  __int64 i; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rax
  int v22; // eax
  __int64 v24; // rax
  char *v25; // rdx
  __int64 v26; // r14
  char v27; // al
  int v28; // eax
  __int64 v29; // rax
  const void *v30; // [rsp+0h] [rbp-40h]
  __int64 j; // [rsp+8h] [rbp-38h]

  v2 = a1 + 96;
  v5 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 176) += 72LL;
  v6 = (v5 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 104) >= v6 + 72 && v5 )
  {
    *(_QWORD *)(a1 + 96) = v6 + 72;
    if ( !v6 )
      goto LABEL_18;
  }
  else
  {
    v6 = sub_9D1E70(a1 + 96, 72, 72, 4);
  }
  LODWORD(v7) = sub_BD3960((__int64)a2);
  *(_BYTE *)(v6 + 52) = 0;
  *(_QWORD *)(v6 + 8) = 0xFFFFFFFD00000006LL;
  *(_DWORD *)(v6 + 32) = v7;
  v7 = (unsigned int)v7;
  *(_QWORD *)(v6 + 16) = 0;
  *(_QWORD *)(v6 + 24) = 0;
  *(_QWORD *)v6 = off_4A20D58;
  *(_QWORD *)(v6 + 36) = 0;
  *(_QWORD *)(v6 + 44) = 0xFFFFFFFF00000000LL;
  *(_QWORD *)(v6 + 56) = 0;
  *(_QWORD *)(v6 + 64) = 0;
  if ( (_DWORD)v7 && (v7 = (unsigned int)v7 - 1LL) != 0 )
  {
    _BitScanReverse64(&v7, v7);
    v8 = 64 - (v7 ^ 0x3F);
    v7 = (int)v8;
    if ( v8 >= *(_DWORD *)(a1 + 200) )
      goto LABEL_24;
  }
  else
  {
    LOBYTE(v8) = 0;
    if ( !*(_DWORD *)(a1 + 200) )
      goto LABEL_24;
  }
  v9 = (_QWORD **)(*(_QWORD *)(a1 + 192) + 8 * v7);
  v10 = *v9;
  if ( *v9 )
  {
    *v9 = (_QWORD *)*v10;
    goto LABEL_8;
  }
LABEL_24:
  v24 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 176) += 8LL << v8;
  v10 = (_QWORD *)((v24 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  v25 = (char *)v10 + (8LL << v8);
  if ( *(_QWORD *)(a1 + 104) >= (unsigned __int64)v25 && v24 )
    *(_QWORD *)(a1 + 96) = v25;
  else
    v10 = (_QWORD *)sub_9D1E70(v2, 8LL << v8, 8LL << v8, 3);
LABEL_8:
  *(_QWORD *)(v6 + 24) = v10;
  v11 = *a2;
  *(_DWORD *)(v6 + 12) = v11 - 29;
  *(_QWORD *)(v6 + 40) = *((_QWORD *)a2 + 1);
  if ( (_BYTE)v11 == 92 )
  {
    v12 = *(_QWORD *)(a1 + 96);
    v13 = *((unsigned int *)a2 + 20);
    v14 = (const void *)*((_QWORD *)a2 + 9);
    v15 = 4 * v13;
    *(_QWORD *)(a1 + 176) += 4 * v13;
    v16 = 4 * v13 + ((v12 + 3) & 0xFFFFFFFFFFFFFFFCLL);
    if ( *(_QWORD *)(a1 + 104) >= v16 && v12 )
    {
      *(_QWORD *)(a1 + 96) = v16;
      v17 = (void *)((v12 + 3) & 0xFFFFFFFFFFFFFFFCLL);
    }
    else
    {
      v30 = v14;
      v29 = sub_9D1E70(v2, 4 * v13, v15, 2);
      v14 = v30;
      v15 = 4 * v13;
      v17 = (void *)v29;
    }
    if ( v15 )
      v17 = memmove(v17, v14, v15);
    *(_QWORD *)(v6 + 56) = v17;
    v10 = *(_QWORD **)(v6 + 24);
    *(_QWORD *)(v6 + 64) = v13;
  }
  for ( i = *((_QWORD *)a2 + 2); i; v10 = *(_QWORD **)(v6 + 24) )
  {
    v19 = *(unsigned int *)(v6 + 36);
    v20 = *(_QWORD *)(i + 24);
    *(_DWORD *)(v6 + 36) = v19 + 1;
    v10[v19] = v20;
    i = *(_QWORD *)(i + 8);
  }
  v21 = *(unsigned int *)(v6 + 36);
  if ( v21 > 1 )
  {
    qsort(v10, (__int64)(8 * v21) >> 3, 8u, (__compar_fn_t)sub_27ABC60);
    if ( !(unsigned __int8)sub_27ABDB0((char *)a2) )
      goto LABEL_19;
    goto LABEL_28;
  }
LABEL_18:
  if ( !(unsigned __int8)sub_27ABDB0((char *)a2) )
    goto LABEL_19;
LABEL_28:
  v26 = *((_QWORD *)a2 + 4);
  for ( j = *((_QWORD *)a2 + 5) + 48LL; j != v26; v26 = *(_QWORD *)(v26 + 8) )
  {
    if ( !v26 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30 <= 0xA )
      break;
    if ( (unsigned __int8)sub_27ABDB0((char *)(v26 - 24)) )
    {
      v27 = *(_BYTE *)(v26 - 24);
      if ( v27 != 61 )
      {
        if ( v27 == 85 )
        {
          if ( (unsigned __int8)sub_B49E20(v26 - 24) )
            continue;
          v27 = *(_BYTE *)(v26 - 24);
        }
        if ( v27 != 34 || !(unsigned __int8)sub_B49E20(v26 - 24) )
        {
          v28 = sub_27B2E30(a1, (unsigned __int8 *)(v26 - 24));
          goto LABEL_39;
        }
      }
    }
  }
  v28 = 0;
LABEL_39:
  *(_DWORD *)(v6 + 48) = v28;
LABEL_19:
  v22 = *a2;
  if ( (unsigned __int8)(v22 - 82) <= 1u )
    *(_DWORD *)(v6 + 12) = *((_WORD *)a2 + 1) & 0x3F | ((v22 - 29) << 8);
  return v6;
}
