// Function: sub_2EDA4D0
// Address: 0x2eda4d0
//
__int64 __fastcall sub_2EDA4D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v10; // rdi
  char v11; // cl
  __int64 v12; // rsi
  __int64 v13; // r11
  unsigned int v14; // edx
  __int64 v15; // r10
  __int64 v16; // r15
  int v17; // r11d
  unsigned __int64 v18; // rdx
  _QWORD *v19; // r14
  unsigned int v20; // esi
  unsigned int v21; // edi
  __int64 v22; // rcx
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rsi
  int v30; // r15d
  int v31; // esi
  unsigned int v32; // edx
  __int64 v33; // rcx
  int v34; // esi
  unsigned int v35; // edx
  __int64 v36; // rcx
  int v37; // r10d
  int v38; // esi
  int v39; // esi
  int v40; // r10d
  _QWORD *v41; // rdi
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]

  result = a1;
  v10 = *(_QWORD *)a2;
  v11 = *(_BYTE *)(a2 + 8) & 1;
  if ( v11 )
  {
    a6 = a2 + 16;
    v12 = *a3;
    v13 = 224;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 3;
    a5 = a6 + 56LL * v14;
    v15 = *(_QWORD *)a5;
    if ( *(_QWORD *)a5 == v12 )
      goto LABEL_3;
    v17 = 3;
  }
  else
  {
    v16 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v16 )
    {
      v18 = *(unsigned int *)(a2 + 8);
      v19 = 0;
      *(_QWORD *)a2 = v10 + 1;
      v20 = ((unsigned int)v18 >> 1) + 1;
      goto LABEL_8;
    }
    a6 = *(_QWORD *)(a2 + 16);
    v12 = *a3;
    v17 = v16 - 1;
    v14 = (v16 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    a5 = a6 + 56LL * v14;
    v15 = *(_QWORD *)a5;
    if ( v12 == *(_QWORD *)a5 )
      goto LABEL_6;
  }
  v30 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v15 == -4096 )
    {
      v18 = *(unsigned int *)(a2 + 8);
      if ( !v19 )
        v19 = (_QWORD *)a5;
      *(_QWORD *)a2 = v10 + 1;
      v20 = ((unsigned int)v18 >> 1) + 1;
      if ( v11 )
      {
        v21 = 12;
        LODWORD(v16) = 4;
LABEL_9:
        v22 = 4 * v20;
        if ( (unsigned int)v22 < v21 )
        {
          v23 = (_DWORD)v16 - *(_DWORD *)(a2 + 12) - v20;
          if ( (unsigned int)v23 > (unsigned int)v16 >> 3 )
          {
LABEL_11:
            v24 = 2 * ((unsigned int)v18 >> 1) + 2;
            *(_DWORD *)(a2 + 8) = v24 | v18 & 1;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a2 + 12);
            v25 = *a3;
            v19[2] = 0x400000000LL;
            *v19 = v25;
            v19[1] = v19 + 3;
            v26 = *(unsigned int *)(a4 + 8);
            if ( (_DWORD)v26 )
            {
              v42 = result;
              sub_2ED1580((__int64)(v19 + 1), (char **)a4, v26, v24, a5, a6);
              result = v42;
            }
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v27 = a2 + 16;
              v28 = 224;
            }
            else
            {
              v27 = *(_QWORD *)(a2 + 16);
              v28 = 56LL * *(unsigned int *)(a2 + 24);
            }
            v29 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v29;
            *(_QWORD *)(result + 24) = v27 + v28;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v44 = result;
          sub_2ED9F30(a2, v16, v18, v23, a5, a6);
          result = v44;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            a6 = a2 + 16;
            v34 = 3;
            goto LABEL_32;
          }
          v39 = *(_DWORD *)(a2 + 24);
          a6 = *(_QWORD *)(a2 + 16);
          if ( v39 )
          {
            v34 = v39 - 1;
LABEL_32:
            v35 = v34 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            a5 = v35;
            v19 = (_QWORD *)(a6 + 56LL * v35);
            v36 = *v19;
            if ( *v19 != *a3 )
            {
              v37 = 1;
              a5 = 0;
              while ( v36 != -4096 )
              {
                if ( v36 == -8192 && !a5 )
                  a5 = (__int64)v19;
                v35 = v34 & (v37 + v35);
                v19 = (_QWORD *)(a6 + 56LL * v35);
                v36 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_29;
                ++v37;
              }
              if ( a5 )
                v19 = (_QWORD *)a5;
            }
LABEL_29:
            LODWORD(v18) = *(_DWORD *)(a2 + 8);
            goto LABEL_11;
          }
LABEL_64:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v43 = result;
        sub_2ED9F30(a2, 2 * v16, v18, v22, a5, a6);
        result = v43;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          a5 = a2 + 16;
          v31 = 3;
        }
        else
        {
          v38 = *(_DWORD *)(a2 + 24);
          a5 = *(_QWORD *)(a2 + 16);
          if ( !v38 )
            goto LABEL_64;
          v31 = v38 - 1;
        }
        a6 = *a3;
        v32 = v31 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (_QWORD *)(a5 + 56LL * v32);
        v33 = *v19;
        if ( *a3 != *v19 )
        {
          v40 = 1;
          v41 = 0;
          while ( v33 != -4096 )
          {
            if ( !v41 && v33 == -8192 )
              v41 = v19;
            v32 = v31 & (v40 + v32);
            v19 = (_QWORD *)(a5 + 56LL * v32);
            v33 = *v19;
            if ( a6 == *v19 )
              goto LABEL_29;
            ++v40;
          }
          if ( v41 )
          {
            LODWORD(v18) = *(_DWORD *)(a2 + 8);
            v19 = v41;
            goto LABEL_11;
          }
        }
        goto LABEL_29;
      }
      LODWORD(v16) = *(_DWORD *)(a2 + 24);
LABEL_8:
      v21 = 3 * v16;
      goto LABEL_9;
    }
    if ( v15 == -8192 && !v19 )
      v19 = (_QWORD *)a5;
    v14 = v17 & (v30 + v14);
    a5 = a6 + 56LL * v14;
    v15 = *(_QWORD *)a5;
    if ( v12 == *(_QWORD *)a5 )
      break;
    ++v30;
  }
  if ( v11 )
  {
    v13 = 224;
  }
  else
  {
    v16 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v13 = 56 * v16;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v10;
  *(_QWORD *)(result + 16) = a5;
  *(_QWORD *)(result + 24) = v13 + a6;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
