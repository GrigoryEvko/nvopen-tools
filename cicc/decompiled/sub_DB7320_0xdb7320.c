// Function: sub_DB7320
// Address: 0xdb7320
//
__int64 __fastcall sub_DB7320(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // r8
  int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // r10
  int v12; // r14d
  unsigned int v13; // ecx
  _QWORD *v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r11
  __int64 v17; // r15
  unsigned int v18; // ecx
  _QWORD *v19; // rdx
  int v20; // edi
  unsigned int v21; // r8d
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rcx
  __int64 v27; // rdi
  int v28; // r15d
  __int64 v29; // r8
  int v30; // esi
  unsigned int v31; // ecx
  __int64 v32; // rdi
  __int64 v33; // r8
  int v34; // esi
  unsigned int v35; // ecx
  __int64 v36; // rdi
  int v37; // r11d
  _QWORD *v38; // r10
  int v39; // esi
  int v40; // esi
  int v41; // r11d
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = *a3;
    v11 = a2 + 16;
    v12 = 3;
    v13 = ((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 3;
    v14 = (_QWORD *)(a2
                   + 16
                   + 56LL
                   * (((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 3));
    v15 = 224;
    v16 = *v14;
    if ( v10 == *v14 )
      goto LABEL_3;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v17 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v19 = 0;
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      goto LABEL_8;
    }
    v10 = *a3;
    v12 = v17 - 1;
    v11 = *(_QWORD *)(a2 + 16);
    v13 = (v17 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v14 = (_QWORD *)(v11 + 56LL * v13);
    v16 = *v14;
    if ( *v14 == *a3 )
      goto LABEL_6;
  }
  v28 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v16 == -4096 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      if ( !v19 )
        v19 = v14;
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      if ( (_BYTE)v9 )
      {
        v21 = 12;
        LODWORD(v17) = 4;
LABEL_9:
        if ( 4 * v20 < v21 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v20 > (unsigned int)v17 >> 3 )
          {
LABEL_11:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a2 + 12);
            v22 = *a3;
            v19[3] = 0;
            v19[2] = 0;
            *((_DWORD *)v19 + 8) = 0;
            *v19 = v22;
            v19[1] = 1;
            v23 = *(_QWORD *)(a4 + 8);
            ++*(_QWORD *)a4;
            v24 = v19[2];
            v19[2] = v23;
            LODWORD(v23) = *(_DWORD *)(a4 + 16);
            *(_QWORD *)(a4 + 8) = v24;
            LODWORD(v24) = *((_DWORD *)v19 + 6);
            *((_DWORD *)v19 + 6) = v23;
            LODWORD(v23) = *(_DWORD *)(a4 + 20);
            *(_DWORD *)(a4 + 16) = v24;
            LODWORD(v24) = *((_DWORD *)v19 + 7);
            *((_DWORD *)v19 + 7) = v23;
            LODWORD(v23) = *(_DWORD *)(a4 + 24);
            *(_DWORD *)(a4 + 20) = v24;
            LODWORD(v24) = *((_DWORD *)v19 + 8);
            *((_DWORD *)v19 + 8) = v23;
            *(_DWORD *)(a4 + 24) = v24;
            *((_WORD *)v19 + 20) = *(_WORD *)(a4 + 32);
            v19[6] = *(_QWORD *)(a4 + 40);
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v25 = a2 + 16;
              v26 = 224;
            }
            else
            {
              v25 = *(_QWORD *)(a2 + 16);
              v26 = 56LL * *(unsigned int *)(a2 + 24);
            }
            v27 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v27;
            *(_QWORD *)(result + 24) = v25 + v26;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v43 = result;
          sub_DB6ED0(a2, v17);
          result = v43;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v33 = a2 + 16;
            v34 = 3;
            goto LABEL_29;
          }
          v40 = *(_DWORD *)(a2 + 24);
          v33 = *(_QWORD *)(a2 + 16);
          if ( v40 )
          {
            v34 = v40 - 1;
LABEL_29:
            v35 = v34 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v19 = (_QWORD *)(v33 + 56LL * v35);
            v36 = *v19;
            if ( *v19 != *a3 )
            {
              v37 = 1;
              v38 = 0;
              while ( v36 != -4096 )
              {
                if ( !v38 && v36 == -8192 )
                  v38 = v19;
                v35 = v34 & (v37 + v35);
                v19 = (_QWORD *)(v33 + 56LL * v35);
                v36 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_26;
                ++v37;
              }
LABEL_32:
              if ( v38 )
                v19 = v38;
              goto LABEL_26;
            }
            goto LABEL_26;
          }
LABEL_59:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v42 = result;
        sub_DB6ED0(a2, 2 * v17);
        result = v42;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v29 = a2 + 16;
          v30 = 3;
        }
        else
        {
          v39 = *(_DWORD *)(a2 + 24);
          v29 = *(_QWORD *)(a2 + 16);
          if ( !v39 )
            goto LABEL_59;
          v30 = v39 - 1;
        }
        v31 = v30 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (_QWORD *)(v29 + 56LL * v31);
        v32 = *v19;
        if ( *a3 != *v19 )
        {
          v41 = 1;
          v38 = 0;
          while ( v32 != -4096 )
          {
            if ( !v38 && v32 == -8192 )
              v38 = v19;
            v31 = v30 & (v41 + v31);
            v19 = (_QWORD *)(v29 + 56LL * v31);
            v32 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_26;
            ++v41;
          }
          goto LABEL_32;
        }
LABEL_26:
        v18 = *(_DWORD *)(a2 + 8);
        goto LABEL_11;
      }
      LODWORD(v17) = *(_DWORD *)(a2 + 24);
LABEL_8:
      v21 = 3 * v17;
      goto LABEL_9;
    }
    if ( !v19 && v16 == -8192 )
      v19 = v14;
    v13 = v12 & (v28 + v13);
    v14 = (_QWORD *)(v11 + 56LL * v13);
    v16 = *v14;
    if ( v10 == *v14 )
      break;
    ++v28;
  }
  if ( (_BYTE)v9 )
  {
    v15 = 224;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v15 = 56 * v17;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v8;
  *(_QWORD *)(result + 16) = v14;
  *(_QWORD *)(result + 24) = v11 + v15;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
