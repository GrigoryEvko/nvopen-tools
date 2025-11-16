// Function: sub_A07210
// Address: 0xa07210
//
__int64 __fastcall sub_A07210(__int64 a1, __int64 a2, int *a3)
{
  __int64 result; // rax
  __int64 v6; // rdi
  char v7; // dl
  int v8; // r10d
  int v9; // r8d
  int *v10; // rcx
  unsigned int v11; // r11d
  __int64 v12; // r9
  __int64 v13; // rsi
  int v14; // r13d
  __int64 v15; // rsi
  char v16; // dl
  __int64 v17; // rsi
  unsigned int v18; // edi
  int v19; // r8d
  unsigned int v20; // r9d
  __int64 v21; // rsi
  __int64 v22; // rdx
  int v23; // r14d
  int *v24; // rsi
  __int64 v25; // r9
  int v26; // edi
  unsigned int v27; // esi
  int v28; // r8d
  __int64 v29; // r9
  int v30; // r8d
  unsigned int v31; // esi
  int v32; // edi
  int v33; // r11d
  int *v34; // r10
  int v35; // edi
  int v36; // r8d
  int v37; // r11d
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  result = a1;
  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(a2 + 8) & 1;
  if ( v7 )
  {
    v8 = *(_DWORD *)(a2 + 16);
    v9 = *a3;
    v10 = (int *)(a2 + 16);
    v11 = 0;
    v12 = a2 + 16;
    v13 = 4;
    v14 = 0;
    if ( v8 == *a3 )
      goto LABEL_3;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v17 )
    {
      v10 = 0;
      *(_QWORD *)a2 = v6 + 1;
      v18 = *(_DWORD *)(a2 + 8);
      v19 = (v18 >> 1) + 1;
      goto LABEL_9;
    }
    v9 = *a3;
    v14 = v17 - 1;
    v12 = *(_QWORD *)(a2 + 16);
    v11 = (v17 - 1) & (37 * *a3);
    v10 = (int *)(v12 + 4LL * v11);
    v8 = *v10;
    if ( *a3 == *v10 )
      goto LABEL_7;
  }
  v23 = 1;
  v24 = 0;
  while ( 1 )
  {
    if ( v8 == -1 )
    {
      if ( v24 )
        v10 = v24;
      *(_QWORD *)a2 = v6 + 1;
      v18 = *(_DWORD *)(a2 + 8);
      v19 = (v18 >> 1) + 1;
      if ( v7 )
      {
        v20 = 3;
        LODWORD(v17) = 1;
LABEL_10:
        if ( v20 > 4 * v19 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v19 > (unsigned int)v17 >> 3 )
          {
LABEL_12:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v10 != -1 )
              --*(_DWORD *)(a2 + 12);
            *v10 = *a3;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v21 = a2 + 16;
              v22 = 4;
            }
            else
            {
              v21 = *(_QWORD *)(a2 + 16);
              v22 = 4LL * *(unsigned int *)(a2 + 24);
            }
            v15 = v22 + v21;
            v6 = *(_QWORD *)a2;
            v16 = 1;
            goto LABEL_4;
          }
          v39 = result;
          sub_A06DF0(a2, v17);
          result = v39;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v29 = a2 + 16;
            v30 = 0;
            goto LABEL_30;
          }
          v36 = *(_DWORD *)(a2 + 24);
          v29 = *(_QWORD *)(a2 + 16);
          if ( v36 )
          {
            v30 = v36 - 1;
LABEL_30:
            v31 = v30 & (37 * *a3);
            v10 = (int *)(v29 + 4LL * v31);
            v32 = *v10;
            if ( *v10 != *a3 )
            {
              v33 = 1;
              v34 = 0;
              while ( v32 != -1 )
              {
                if ( v32 == -2 && !v34 )
                  v34 = v10;
                v31 = v30 & (v33 + v31);
                v10 = (int *)(v29 + 4LL * v31);
                v32 = *v10;
                if ( *a3 == *v10 )
                  goto LABEL_27;
                ++v33;
              }
LABEL_33:
              if ( v34 )
                v10 = v34;
              goto LABEL_27;
            }
            goto LABEL_27;
          }
LABEL_60:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v38 = result;
        sub_A06DF0(a2, 2 * v17);
        result = v38;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v25 = a2 + 16;
          v26 = 0;
        }
        else
        {
          v35 = *(_DWORD *)(a2 + 24);
          v25 = *(_QWORD *)(a2 + 16);
          if ( !v35 )
            goto LABEL_60;
          v26 = v35 - 1;
        }
        v27 = v26 & (37 * *a3);
        v10 = (int *)(v25 + 4LL * v27);
        v28 = *v10;
        if ( *a3 != *v10 )
        {
          v37 = 1;
          v34 = 0;
          while ( v28 != -1 )
          {
            if ( !v34 && v28 == -2 )
              v34 = v10;
            v27 = v26 & (v37 + v27);
            v10 = (int *)(v25 + 4LL * v27);
            v28 = *v10;
            if ( *a3 == *v10 )
              goto LABEL_27;
            ++v37;
          }
          goto LABEL_33;
        }
LABEL_27:
        v18 = *(_DWORD *)(a2 + 8);
        goto LABEL_12;
      }
      LODWORD(v17) = *(_DWORD *)(a2 + 24);
LABEL_9:
      v20 = 3 * v17;
      goto LABEL_10;
    }
    if ( v8 == -2 && !v24 )
      v24 = v10;
    v11 = v14 & (v23 + v11);
    v10 = (int *)(v12 + 4LL * v11);
    v8 = *v10;
    if ( *v10 == v9 )
      break;
    ++v23;
  }
  if ( v7 )
  {
    v13 = 4;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_7:
    v13 = 4 * v17;
  }
LABEL_3:
  v15 = v12 + v13;
  v16 = 0;
LABEL_4:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v6;
  *(_QWORD *)(result + 16) = v10;
  *(_QWORD *)(result + 24) = v15;
  *(_BYTE *)(result + 32) = v16;
  return result;
}
