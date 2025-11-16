// Function: sub_1DAD0A0
// Address: 0x1dad0a0
//
void __fastcall sub_1DAD0A0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned int v10; // edi
  __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 *v15; // r9
  __int64 *v16; // rax
  unsigned int v17; // eax
  _QWORD *v18; // rcx
  _QWORD *v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  unsigned int v25; // esi
  __int64 v26; // r8
  __int64 v27; // rcx
  _QWORD *v28; // rdx
  _QWORD *v29; // rax

  v8 = *(_QWORD *)a1;
  if ( *(_DWORD *)(*(_QWORD *)a1 + 80LL) )
    goto LABEL_18;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *(_DWORD *)(v8 + 84);
  v11 = v9 + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v12 = *(_DWORD *)(v11 + 12);
  if ( v12 )
  {
    v13 = v12 - 1;
    if ( ((a4 ^ *(_DWORD *)(v8 + 4 * v13 + 64)) & 0x7FFFFFFF) == 0
      && ((*(_BYTE *)(v8 + 4 * (v13 + 16) + 3) ^ HIBYTE(a4)) & 0x80u) == 0 )
    {
      v14 = 16 * v13;
      v15 = (__int64 *)(v8 + v14 + 8);
      if ( *v15 == a2 )
      {
        *(_DWORD *)(v11 + 12) = v12 - 1;
        if ( v10 == v12
          || ((a4 ^ *(_DWORD *)(v8 + 4LL * v12 + 64)) & 0x7FFFFFFF) != 0
          || ((*(_BYTE *)(v8 + 4 * (v12 + 16LL) + 3) ^ HIBYTE(a4)) & 0x80u) != 0
          || (v24 = (_QWORD *)(v8 + 16LL * v12), *v24 != a3) )
        {
          *v15 = a3;
        }
        else
        {
          v25 = v12 + 1;
          for ( *(_QWORD *)(v8 + v14 + 8) = v24[1];
                v10 != v25;
                *(_DWORD *)(v8 + 4 * v27 + 64) = *(_DWORD *)(v8 + 4 * v26 + 64) )
          {
            v26 = v25;
            v27 = v25++ - 1;
            v28 = (_QWORD *)(v8 + 16 * v26);
            v29 = (_QWORD *)(v8 + 16 * v27);
            *v29 = *v28;
            v29[1] = v28[1];
          }
          --v10;
        }
        goto LABEL_14;
      }
    }
    if ( v12 == 4 )
      goto LABEL_17;
  }
  if ( v10 == v12 )
  {
    v22 = v10++;
    v23 = 16 * v22;
    *(_QWORD *)(v8 + v23) = a2;
    *(_QWORD *)(v8 + v23 + 8) = a3;
    *(_DWORD *)(v8 + 4 * v22 + 64) = a4;
LABEL_14:
    if ( v10 <= 4 )
    {
      *(_DWORD *)(v8 + 84) = v10;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v10;
      return;
    }
    v12 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 4);
    goto LABEL_17;
  }
  if ( ((a4 ^ *(_DWORD *)(v8 + 4LL * v12 + 64)) & 0x7FFFFFFF) == 0
    && ((*(_BYTE *)(v8 + 4 * (v12 + 16LL) + 3) ^ HIBYTE(a4)) & 0x80u) == 0 )
  {
    v16 = (__int64 *)(v8 + 16LL * v12);
    if ( *v16 == a3 )
    {
      *v16 = a2;
      goto LABEL_14;
    }
  }
  v17 = v10 - 1;
  if ( v10 != 4 )
  {
    do
    {
      v18 = (_QWORD *)(v8 + 16LL * v17);
      v19 = (_QWORD *)(v8 + 16LL * (v17 + 1));
      *v19 = *v18;
      v19[1] = v18[1];
      *(_DWORD *)(v8 + 4LL * (v17 + 1) + 64) = *(_DWORD *)(v8 + 4LL * v17 + 64);
      LODWORD(v19) = v17--;
    }
    while ( v12 != (_DWORD)v19 );
    ++v10;
    v20 = 16LL * v12;
    *(_QWORD *)(v8 + v20) = a2;
    *(_QWORD *)(v8 + v20 + 8) = a3;
    *(_DWORD *)(v8 + 4LL * v12 + 64) = a4;
    goto LABEL_14;
  }
LABEL_17:
  v21 = sub_1DA9830(v8, v12);
  sub_3945C20(a1 + 8, v8 + 8, *(unsigned int *)(v8 + 84), v21);
LABEL_18:
  sub_1DAC9A0(a1, a2, a3, a4);
}
