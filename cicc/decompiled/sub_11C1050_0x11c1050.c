// Function: sub_11C1050
// Address: 0x11c1050
//
__int64 __fastcall sub_11C1050(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // r8
  char v7; // cl
  __int64 v8; // r9
  __int64 v9; // r10
  int v10; // r13d
  unsigned int v11; // edx
  __int64 *v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // r11
  __int64 v15; // rsi
  char v16; // dl
  __int64 v17; // rsi
  unsigned int v18; // edx
  int v19; // r8d
  unsigned int v20; // r9d
  __int64 v21; // rsi
  __int64 v22; // rdx
  int v23; // r14d
  __int64 *v24; // rsi
  __int64 v25; // r8
  int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // r9
  __int64 v29; // r8
  int v30; // ecx
  unsigned int v31; // edx
  __int64 v32; // r9
  int v33; // r11d
  __int64 *v34; // r10
  int v35; // ecx
  int v36; // ecx
  int v37; // r11d
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  result = a1;
  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(a2 + 8) & 1;
  if ( v7 )
  {
    v8 = *a3;
    v9 = a2 + 16;
    v10 = 3;
    v11 = ((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 3;
    v12 = (__int64 *)(a2 + 16 + 8LL * v11);
    v13 = 32;
    v14 = *v12;
    if ( *v12 == v8 )
      goto LABEL_3;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v17 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v12 = 0;
      *(_QWORD *)a2 = v6 + 1;
      v19 = (v18 >> 1) + 1;
      goto LABEL_9;
    }
    v8 = *a3;
    v10 = v17 - 1;
    v9 = *(_QWORD *)(a2 + 16);
    v11 = (v17 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v12 = (__int64 *)(v9 + 8LL * v11);
    v14 = *v12;
    if ( *v12 == v8 )
      goto LABEL_7;
  }
  v23 = 1;
  v24 = 0;
  while ( 1 )
  {
    if ( v14 == -4096 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      if ( v24 )
        v12 = v24;
      *(_QWORD *)a2 = v6 + 1;
      v19 = (v18 >> 1) + 1;
      if ( v7 )
      {
        v20 = 12;
        LODWORD(v17) = 4;
LABEL_10:
        if ( v20 > 4 * v19 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v19 > (unsigned int)v17 >> 3 )
          {
LABEL_12:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v12 != -4096 )
              --*(_DWORD *)(a2 + 12);
            *v12 = *a3;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v21 = a2 + 16;
              v22 = 32;
            }
            else
            {
              v21 = *(_QWORD *)(a2 + 16);
              v22 = 8LL * *(unsigned int *)(a2 + 24);
            }
            v15 = v22 + v21;
            v6 = *(_QWORD *)a2;
            v16 = 1;
            goto LABEL_4;
          }
          v39 = result;
          sub_11C0C40(a2, v17);
          result = v39;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v29 = a2 + 16;
            v30 = 3;
            goto LABEL_30;
          }
          v36 = *(_DWORD *)(a2 + 24);
          v29 = *(_QWORD *)(a2 + 16);
          if ( v36 )
          {
            v30 = v36 - 1;
LABEL_30:
            v31 = v30 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v12 = (__int64 *)(v29 + 8LL * v31);
            v32 = *v12;
            if ( *v12 != *a3 )
            {
              v33 = 1;
              v34 = 0;
              while ( v32 != -4096 )
              {
                if ( v32 == -8192 && !v34 )
                  v34 = v12;
                v31 = v30 & (v33 + v31);
                v12 = (__int64 *)(v29 + 8LL * v31);
                v32 = *v12;
                if ( *a3 == *v12 )
                  goto LABEL_27;
                ++v33;
              }
LABEL_33:
              if ( v34 )
                v12 = v34;
              goto LABEL_27;
            }
            goto LABEL_27;
          }
LABEL_60:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v38 = result;
        sub_11C0C40(a2, 2 * v17);
        result = v38;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v25 = a2 + 16;
          v26 = 3;
        }
        else
        {
          v35 = *(_DWORD *)(a2 + 24);
          v25 = *(_QWORD *)(a2 + 16);
          if ( !v35 )
            goto LABEL_60;
          v26 = v35 - 1;
        }
        v27 = v26 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v12 = (__int64 *)(v25 + 8LL * v27);
        v28 = *v12;
        if ( *a3 != *v12 )
        {
          v37 = 1;
          v34 = 0;
          while ( v28 != -4096 )
          {
            if ( !v34 && v28 == -8192 )
              v34 = v12;
            v27 = v26 & (v37 + v27);
            v12 = (__int64 *)(v25 + 8LL * v27);
            v28 = *v12;
            if ( *a3 == *v12 )
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
    if ( v14 == -8192 && !v24 )
      v24 = v12;
    v11 = v10 & (v23 + v11);
    v12 = (__int64 *)(v9 + 8LL * v11);
    v14 = *v12;
    if ( *v12 == v8 )
      break;
    ++v23;
  }
  if ( v7 )
  {
    v13 = 32;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_7:
    v13 = 8 * v17;
  }
LABEL_3:
  v15 = v9 + v13;
  v16 = 0;
LABEL_4:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v6;
  *(_QWORD *)(result + 16) = v12;
  *(_QWORD *)(result + 24) = v15;
  *(_BYTE *)(result + 32) = v16;
  return result;
}
