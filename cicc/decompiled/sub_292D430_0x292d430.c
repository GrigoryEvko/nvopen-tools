// Function: sub_292D430
// Address: 0x292d430
//
__int64 __fastcall sub_292D430(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 result; // rax
  char v8; // dl
  __int64 v9; // rsi
  __int64 v10; // r8
  int v11; // edx
  __int64 v12; // r9
  __int64 *v13; // rcx
  __int64 v14; // r15
  __int64 v15; // r15
  int v16; // r11d
  unsigned int v17; // r10d
  unsigned int v18; // ecx
  __int64 *v19; // rdi
  int v20; // esi
  unsigned int v21; // r8d
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // r14
  int v29; // r15d
  __int64 v30; // r8
  int v31; // esi
  unsigned int v32; // edx
  __int64 v33; // r9
  __int64 v34; // r9
  int v35; // esi
  unsigned int v36; // edx
  __int64 v37; // r8
  int v38; // r11d
  __int64 *v39; // r10
  int v40; // esi
  int v41; // esi
  int v42; // r11d
  __int64 v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(_BYTE *)(a2 + 8);
  v9 = *(_QWORD *)(a2 + 16);
  v10 = *(_QWORD *)a2;
  v11 = v8 & 1;
  if ( v11 )
  {
    v12 = *a3;
    v13 = (__int64 *)(a2 + 16);
    if ( v9 == *a3 )
    {
      v9 = a2 + 16;
      v14 = 32;
      goto LABEL_4;
    }
    v28 = a2 + 16;
    v17 = 0;
    v16 = 0;
  }
  else
  {
    v15 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v15 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v19 = 0;
      *(_QWORD *)a2 = v10 + 1;
      v20 = (v18 >> 1) + 1;
      goto LABEL_9;
    }
    v12 = *a3;
    v16 = v15 - 1;
    v17 = (v15 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v13 = (__int64 *)(v9 + 32LL * v17);
    if ( *a3 == *v13 )
      goto LABEL_7;
    v28 = v9;
    v9 = *v13;
  }
  v29 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v9 == -4096 )
    {
      if ( !v19 )
        v19 = v13;
      v18 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v10 + 1;
      v20 = (v18 >> 1) + 1;
      if ( (_BYTE)v11 )
      {
        v21 = 3;
        LODWORD(v15) = 1;
LABEL_10:
        if ( v21 > 4 * v20 )
        {
          if ( (int)v15 - *(_DWORD *)(a2 + 12) - v20 > (unsigned int)v15 >> 3 )
          {
LABEL_12:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a2 + 12);
            *v19 = *a3;
            v22 = *a4;
            *a4 = 0;
            v19[1] = v22;
            v23 = a4[1];
            a4[1] = 0;
            v19[2] = v23;
            v24 = a4[2];
            a4[2] = 0;
            v19[3] = v24;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v25 = a2 + 16;
              v26 = 32;
            }
            else
            {
              v25 = *(_QWORD *)(a2 + 16);
              v26 = 32LL * *(unsigned int *)(a2 + 24);
            }
            v27 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v27;
            *(_QWORD *)(result + 24) = v25 + v26;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v44 = result;
          sub_292D1D0(a2, v15);
          result = v44;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v34 = a2 + 16;
            v35 = 0;
            goto LABEL_31;
          }
          v41 = *(_DWORD *)(a2 + 24);
          v34 = *(_QWORD *)(a2 + 16);
          if ( v41 )
          {
            v35 = v41 - 1;
LABEL_31:
            v36 = v35 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v19 = (__int64 *)(v34 + 32LL * v36);
            v37 = *v19;
            if ( *v19 != *a3 )
            {
              v38 = 1;
              v39 = 0;
              while ( v37 != -4096 )
              {
                if ( v37 == -8192 && !v39 )
                  v39 = v19;
                v36 = v35 & (v38 + v36);
                v19 = (__int64 *)(v34 + 32LL * v36);
                v37 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_28;
                ++v38;
              }
LABEL_34:
              if ( v39 )
                v19 = v39;
              goto LABEL_28;
            }
            goto LABEL_28;
          }
LABEL_62:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v43 = result;
        sub_292D1D0(a2, 2 * v15);
        result = v43;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v30 = a2 + 16;
          v31 = 0;
        }
        else
        {
          v40 = *(_DWORD *)(a2 + 24);
          v30 = *(_QWORD *)(a2 + 16);
          if ( !v40 )
            goto LABEL_62;
          v31 = v40 - 1;
        }
        v32 = v31 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (__int64 *)(v30 + 32LL * v32);
        v33 = *v19;
        if ( *a3 != *v19 )
        {
          v42 = 1;
          v39 = 0;
          while ( v33 != -4096 )
          {
            if ( !v39 && v33 == -8192 )
              v39 = v19;
            v32 = v31 & (v42 + v32);
            v19 = (__int64 *)(v30 + 32LL * v32);
            v33 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_28;
            ++v42;
          }
          goto LABEL_34;
        }
LABEL_28:
        v18 = *(_DWORD *)(a2 + 8);
        goto LABEL_12;
      }
      LODWORD(v15) = *(_DWORD *)(a2 + 24);
LABEL_9:
      v21 = 3 * v15;
      goto LABEL_10;
    }
    if ( !v19 && v9 == -8192 )
      v19 = v13;
    v17 = v16 & (v29 + v17);
    v13 = (__int64 *)(v28 + 32LL * v17);
    v9 = *v13;
    if ( v12 == *v13 )
      break;
    ++v29;
  }
  if ( (_BYTE)v11 )
  {
    v9 = v28;
    v14 = 32;
  }
  else
  {
    v15 = *(unsigned int *)(a2 + 24);
    v9 = v28;
LABEL_7:
    v14 = 32 * v15;
  }
LABEL_4:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v10;
  *(_QWORD *)(result + 16) = v13;
  *(_QWORD *)(result + 24) = v9 + v14;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
