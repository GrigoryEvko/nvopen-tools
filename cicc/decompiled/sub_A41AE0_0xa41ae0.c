// Function: sub_A41AE0
// Address: 0xa41ae0
//
void __fastcall sub_A41AE0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 *a4, __m128i a5)
{
  unsigned int v9; // esi
  __int64 v10; // r9
  __int64 v11; // r8
  int v12; // r10d
  _QWORD *v13; // rax
  unsigned int v14; // edi
  __int64 *v15; // rdx
  __int64 v16; // rcx
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // r8
  _QWORD *v21; // rbx
  _BYTE *v22; // rdi
  __int64 v23; // rax
  int v24; // eax
  int v25; // esi
  __int64 v26; // rdi
  unsigned int v27; // edx
  int v28; // ecx
  __int64 v29; // r8
  int v30; // r10d
  int v31; // ecx
  int v32; // eax
  int v33; // edx
  __int64 v34; // rdi
  _QWORD *v35; // r8
  unsigned int v36; // ebx
  __int64 v37; // rsi
  _QWORD *v38; // [rsp+8h] [rbp-38h]

  v9 = *(_DWORD *)(a3 + 24);
  if ( !v9 )
    goto LABEL_14;
LABEL_2:
  v10 = v9 - 1;
  v11 = *(_QWORD *)(a3 + 8);
  v12 = 1;
  v13 = 0;
  v14 = v10 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v15 = (__int64 *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( a1 != *v15 )
  {
    while ( 1 )
    {
      if ( v16 == -4096 )
      {
        v31 = *(_DWORD *)(a3 + 16);
        if ( !v13 )
          v13 = v15;
        ++*(_QWORD *)a3;
        v28 = v31 + 1;
        if ( 4 * v28 >= 3 * v9 )
        {
          while ( 1 )
          {
            sub_A41580(a3, 2 * v9);
            v24 = *(_DWORD *)(a3 + 24);
            if ( !v24 )
              break;
            v25 = v24 - 1;
            v26 = *(_QWORD *)(a3 + 8);
            v27 = (v24 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v28 = *(_DWORD *)(a3 + 16) + 1;
            v13 = (_QWORD *)(v26 + 16LL * v27);
            v29 = *v13;
            if ( a1 != *v13 )
            {
              v30 = 1;
              v10 = 0;
              while ( v29 != -4096 )
              {
                if ( !v10 && v29 == -8192 )
                  v10 = (__int64)v13;
                v27 = v25 & (v30 + v27);
                v13 = (_QWORD *)(v26 + 16LL * v27);
                v29 = *v13;
                if ( a1 == *v13 )
                  goto LABEL_37;
                ++v30;
              }
              if ( v10 )
                v13 = (_QWORD *)v10;
            }
LABEL_37:
            *(_DWORD *)(a3 + 16) = v28;
            if ( *v13 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v13 = a1;
            v17 = v13 + 1;
            *(_DWORD *)v17 = 0;
            *((_BYTE *)v17 + 4) = 0;
LABEL_4:
            *((_BYTE *)v17 + 4) = 1;
            v18 = *(_QWORD *)(a1 + 16);
            if ( v18 && *(_QWORD *)(v18 + 8) )
              sub_A3FDF0(a1, a2, *(_DWORD *)v17, a3, a4, v10, a5);
            if ( *(_BYTE *)a1 > 0x15u || (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0 )
              return;
            v19 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(a1 + 7) & 0x40) == 0 )
            {
              v20 = (_QWORD *)a1;
              v21 = (_QWORD *)(a1 - v19 * 8);
              goto LABEL_9;
            }
            v21 = *(_QWORD **)(a1 - 8);
            v22 = (_BYTE *)*v21;
            v20 = &v21[v19];
            if ( *(_BYTE *)*v21 <= 0x15u )
            {
LABEL_24:
              v38 = v20;
              sub_A41AE0(v22, a2, a3, a4);
              v20 = v38;
            }
            while ( 1 )
            {
              v21 += 4;
              if ( v20 == v21 )
                break;
LABEL_9:
              v22 = (_BYTE *)*v21;
              if ( *(_BYTE *)*v21 <= 0x15u )
                goto LABEL_24;
            }
            if ( *(_BYTE *)a1 != 5 || *(_WORD *)(a1 + 2) != 63 )
              return;
            v23 = sub_AC3600(a1);
            v9 = *(_DWORD *)(a3 + 24);
            a1 = v23;
            if ( v9 )
              goto LABEL_2;
LABEL_14:
            ++*(_QWORD *)a3;
          }
        }
        else
        {
          if ( v9 - *(_DWORD *)(a3 + 20) - v28 > v9 >> 3 )
            goto LABEL_37;
          sub_A41580(a3, v9);
          v32 = *(_DWORD *)(a3 + 24);
          if ( v32 )
          {
            v33 = v32 - 1;
            v34 = *(_QWORD *)(a3 + 8);
            v35 = 0;
            v36 = (v32 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
            v10 = 1;
            v28 = *(_DWORD *)(a3 + 16) + 1;
            v13 = (_QWORD *)(v34 + 16LL * v36);
            v37 = *v13;
            if ( a1 != *v13 )
            {
              while ( v37 != -4096 )
              {
                if ( v37 == -8192 && !v35 )
                  v35 = v13;
                v36 = v33 & (v10 + v36);
                v13 = (_QWORD *)(v34 + 16LL * v36);
                v37 = *v13;
                if ( a1 == *v13 )
                  goto LABEL_37;
                v10 = (unsigned int)(v10 + 1);
              }
              if ( v35 )
                v13 = v35;
            }
            goto LABEL_37;
          }
        }
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
      if ( v16 == -8192 && !v13 )
        v13 = v15;
      v14 = v10 & (v12 + v14);
      v15 = (__int64 *)(v11 + 16LL * v14);
      v16 = *v15;
      if ( a1 == *v15 )
        break;
      ++v12;
    }
  }
  v17 = v15 + 1;
  if ( !*((_BYTE *)v15 + 12) )
    goto LABEL_4;
}
