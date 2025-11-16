// Function: sub_18F21F0
// Address: 0x18f21f0
//
__int64 __fastcall sub_18F21F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  int v7; // r14d
  __int64 *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  int v11; // r8d
  __int64 v12; // r9
  __int64 v13; // r12
  char v15; // dl
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // esi
  unsigned int v19; // eax
  _QWORD *v20; // rdi
  int v21; // r11d
  _QWORD *v22; // r10
  unsigned int v23; // eax
  int v24; // ecx
  unsigned int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rdi
  int v28; // edx
  unsigned int v29; // eax
  __int64 v30; // r11
  __int64 v31; // rdi
  int v32; // edx
  unsigned int v33; // eax
  __int64 v34; // r11
  int v35; // esi
  _QWORD *v36; // rcx
  int v37; // edx
  int v38; // edx
  int v39; // esi
  const void *v40; // [rsp+0h] [rbp-40h]
  __int64 v41; // [rsp+8h] [rbp-38h]

  sub_1AEAA40(a1);
  v40 = (const void *)(a2 + 160);
  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
  {
    v41 = a3;
    v5 = 0;
    v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    while ( 1 )
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v8 = (__int64 *)(*(_QWORD *)(a1 - 8) + 24 * v5);
      else
        v8 = (__int64 *)(a1 + 24 * (v5 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      v13 = *v8;
      if ( *v8 )
      {
        v9 = v8[1];
        v10 = v8[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v10 = v9;
        if ( v9 )
          *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
      }
      *v8 = 0;
      if ( *(_QWORD *)(v13 + 8) || a1 == v13 || *(_BYTE *)(v13 + 16) <= 0x17u || !(unsigned __int8)sub_1AE9990(v13, v41) )
        goto LABEL_11;
      v15 = *(_BYTE *)(a2 + 8) & 1;
      if ( v15 )
      {
        v17 = a2 + 16;
        v18 = 15;
      }
      else
      {
        v16 = *(_DWORD *)(a2 + 24);
        v17 = *(_QWORD *)(a2 + 16);
        if ( !v16 )
        {
          v23 = *(_DWORD *)(a2 + 8);
          ++*(_QWORD *)a2;
          v22 = 0;
          v24 = (v23 >> 1) + 1;
          goto LABEL_27;
        }
        v18 = v16 - 1;
      }
      v19 = v18 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v20 = (_QWORD *)(v17 + 8LL * v19);
      v12 = *v20;
      if ( *v20 != v13 )
        break;
LABEL_11:
      if ( v7 == (_DWORD)++v5 )
        goto LABEL_14;
    }
    v21 = 1;
    v22 = 0;
    while ( v12 != -8 )
    {
      if ( v22 || v12 != -16 )
        v20 = v22;
      v19 = v18 & (v21 + v19);
      v12 = *(_QWORD *)(v17 + 8LL * v19);
      v11 = v17 + 8 * v19;
      if ( v12 == v13 )
        goto LABEL_11;
      ++v21;
      v22 = v20;
      v20 = (_QWORD *)(v17 + 8LL * v19);
    }
    v23 = *(_DWORD *)(a2 + 8);
    if ( !v22 )
      v22 = v20;
    ++*(_QWORD *)a2;
    v24 = (v23 >> 1) + 1;
    if ( v15 )
    {
      v25 = 48;
      v16 = 16;
      goto LABEL_28;
    }
    v16 = *(_DWORD *)(a2 + 24);
LABEL_27:
    v25 = 3 * v16;
LABEL_28:
    if ( 4 * v24 >= v25 )
    {
      sub_18F1E30(a2, 2 * v16);
      if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
      {
        v27 = a2 + 16;
        v28 = 15;
      }
      else
      {
        v37 = *(_DWORD *)(a2 + 24);
        v27 = *(_QWORD *)(a2 + 16);
        if ( !v37 )
          goto LABEL_68;
        v28 = v37 - 1;
      }
      v29 = v28 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v22 = (_QWORD *)(v27 + 8LL * v29);
      v30 = *v22;
      if ( *v22 == v13 )
        goto LABEL_38;
      v39 = 1;
      v36 = 0;
      while ( v30 != -8 )
      {
        if ( !v36 && v30 == -16 )
          v36 = v22;
        LODWORD(v12) = v39 + 1;
        v29 = v28 & (v29 + v39);
        v22 = (_QWORD *)(v27 + 8LL * v29);
        v30 = *v22;
        if ( *v22 == v13 )
          goto LABEL_38;
        ++v39;
      }
    }
    else
    {
      if ( v16 - *(_DWORD *)(a2 + 12) - v24 > v16 >> 3 )
      {
LABEL_30:
        *(_DWORD *)(a2 + 8) = (2 * (v23 >> 1) + 2) | v23 & 1;
        if ( *v22 != -8 )
          --*(_DWORD *)(a2 + 12);
        *v22 = v13;
        v26 = *(unsigned int *)(a2 + 152);
        if ( (unsigned int)v26 >= *(_DWORD *)(a2 + 156) )
        {
          sub_16CD150(a2 + 144, v40, 0, 8, v11, v12);
          v26 = *(unsigned int *)(a2 + 152);
        }
        *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v26) = v13;
        ++*(_DWORD *)(a2 + 152);
        goto LABEL_11;
      }
      sub_18F1E30(a2, v16);
      if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
      {
        v31 = a2 + 16;
        v32 = 15;
      }
      else
      {
        v38 = *(_DWORD *)(a2 + 24);
        v31 = *(_QWORD *)(a2 + 16);
        if ( !v38 )
        {
LABEL_68:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v32 = v38 - 1;
      }
      v33 = v32 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v22 = (_QWORD *)(v31 + 8LL * v33);
      v34 = *v22;
      if ( *v22 == v13 )
      {
LABEL_38:
        v23 = *(_DWORD *)(a2 + 8);
        goto LABEL_30;
      }
      v35 = 1;
      v36 = 0;
      while ( v34 != -8 )
      {
        if ( !v36 && v34 == -16 )
          v36 = v22;
        LODWORD(v12) = v35 + 1;
        v33 = v32 & (v33 + v35);
        v22 = (_QWORD *)(v31 + 8LL * v33);
        v34 = *v22;
        if ( *v22 == v13 )
          goto LABEL_38;
        ++v35;
      }
    }
    if ( v36 )
      v22 = v36;
    goto LABEL_38;
  }
LABEL_14:
  sub_15F20C0((_QWORD *)a1);
  return 1;
}
