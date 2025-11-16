// Function: sub_1DAAD40
// Address: 0x1daad40
//
void __fastcall sub_1DAAD40(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // r9
  __int64 v5; // r8
  __int64 v6; // rsi
  int v7; // r8d
  __int64 v8; // rcx
  __int64 v9; // r11
  unsigned int v10; // edx
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r11
  unsigned int v16; // edi
  __int64 v17; // rsi
  unsigned __int64 v18; // r8
  __int64 j; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rbx
  unsigned int v26; // edx
  unsigned int v27; // edi
  unsigned int v28; // r8d
  __int64 v29; // r9
  __int64 v30; // rax
  int v31; // eax
  unsigned __int64 v32; // r14
  bool v33; // zf
  __int64 i; // rcx
  unsigned int v35; // edi
  __int64 v36; // r10
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rax

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = a2 >> 1;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v5 & 3;
  v8 = *(unsigned int *)(a1 + 16);
  v9 = v6 + 16 * v8 - 16;
  v10 = v7 | *(_DWORD *)(v4 + 24);
  v11 = *(_QWORD *)v9;
  v12 = *(_QWORD *)(*(_QWORD *)v9 + 16LL * (unsigned int)(*(_DWORD *)(v9 + 8) - 1) + 8);
  if ( (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v12 >> 1) & 3) > v10 )
  {
    for ( i = *(unsigned int *)(v9 + 12);
          v10 >= (*(_DWORD *)((*(_QWORD *)(v11 + 16 * i + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                | (unsigned int)(*(__int64 *)(v11 + 16 * i + 8) >> 1) & 3);
          i = (unsigned int)(i + 1) )
    {
      ;
    }
    *(_DWORD *)(v9 + 12) = i;
  }
  else
  {
    *(_DWORD *)(a1 + 16) = v8 - 1;
    if ( (_DWORD)v8 == 2 )
    {
      v24 = *(unsigned int *)(v6 + 12);
      v25 = *(_QWORD *)a1;
      v26 = *(_DWORD *)(v6 + 12);
    }
    else
    {
      v13 = (unsigned int)(v8 - 3);
      v14 = v6 + 16 * v13;
      if ( (_DWORD)v13 )
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(*(_QWORD *)v14 + 8LL * *(unsigned int *)(v14 + 12) + 96);
          v16 = v7 | *(_DWORD *)(v4 + 24);
          if ( (*(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v15 >> 1) & 3) > v16 )
            break;
          --*(_DWORD *)(a1 + 16);
          v14 -= 16;
          LODWORD(v13) = v13 - 1;
          if ( !(_DWORD)v13 )
            goto LABEL_28;
        }
        v17 = 16LL * (unsigned int)(v13 + 1) + v6;
        v18 = *(_QWORD *)v17;
        for ( j = *(unsigned int *)(v17 + 12); ; j = (unsigned int)(v20 + 1) )
        {
          v20 = j;
          v21 = *(_QWORD *)(v18 + 8 * j + 96);
          v22 = v21 >> 1;
          v23 = v21 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v16 < (*(_DWORD *)(v23 + 24) | (unsigned int)(v22 & 3)) )
            break;
        }
        *(_DWORD *)(v17 + 12) = v20;
        goto LABEL_11;
      }
LABEL_28:
      v24 = *(unsigned int *)(v6 + 12);
      v25 = *(_QWORD *)a1;
      v26 = *(_DWORD *)(v6 + 12);
      v35 = v7 | *(_DWORD *)(v4 + 24);
      v36 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v24 + 40);
      if ( (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v36 >> 1) & 3) > v35 )
      {
        v18 = *(_QWORD *)(v6 + 16);
        v20 = *(unsigned int *)(v6 + 28);
        v37 = *(_QWORD *)(v18 + 8 * v20 + 96);
        v23 = v37 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v35 >= (*(_DWORD *)((v37 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v37 >> 1) & 3) )
        {
          do
          {
            v20 = (unsigned int)(v20 + 1);
            v38 = *(_QWORD *)(v18 + 8 * v20 + 96);
            v39 = v38 >> 1;
            v23 = v38 & 0xFFFFFFFFFFFFFFF8LL;
          }
          while ( v35 >= (*(_DWORD *)(v23 + 24) | (unsigned int)(v39 & 3)) );
        }
        *(_DWORD *)(v6 + 28) = v20;
LABEL_11:
        sub_1DAAA30(a1, a2, v20, v23, v18);
        return;
      }
    }
    v27 = *(_DWORD *)(v25 + 84);
    if ( v27 != v26 )
    {
      v28 = *(_DWORD *)(v4 + 24) | v7;
      while ( 1 )
      {
        v29 = *(_QWORD *)(v25 + 8 + 8 * v24 + 32);
        v30 = v29 >> 1;
        v4 = v29 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_DWORD *)(v4 + 24) | (unsigned int)(v30 & 3)) > v28 )
          break;
        if ( v27 == ++v26 )
        {
          v24 = v27;
          break;
        }
        v24 = v26;
      }
    }
    v20 = *(unsigned int *)(v25 + 80);
    v31 = *(_DWORD *)(a1 + 20);
    v18 = a1 + 8;
    *(_DWORD *)(a1 + 16) = 0;
    if ( (_DWORD)v20 )
      v25 += 8;
    v23 = v24 << 32;
    v32 = v23 | v27;
    if ( !v31 )
    {
      sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v18, v4);
      v6 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16);
    }
    *(_QWORD *)v6 = v25;
    *(_QWORD *)(v6 + 8) = v32;
    v33 = (*(_DWORD *)(a1 + 16))++ == -1;
    if ( !v33 && *(_DWORD *)(*(_QWORD *)(a1 + 8) + 12LL) < *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) )
      goto LABEL_11;
  }
}
