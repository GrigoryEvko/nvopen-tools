// Function: sub_2B564D0
// Address: 0x2b564d0
//
__int64 __fastcall sub_2B564D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // rdi
  char v9; // cl
  __int64 v10; // r10
  __int64 v11; // rsi
  __int64 v12; // r9
  int v13; // r14d
  unsigned int v14; // edx
  __int64 *v15; // r8
  __int64 v16; // r11
  __int64 v17; // r15
  unsigned int v18; // edx
  __int64 *v19; // r9
  int v20; // esi
  unsigned int v21; // edi
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rsi
  int v26; // r15d
  __int64 v27; // r8
  int v28; // ecx
  unsigned int v29; // edx
  __int64 v30; // rsi
  __int64 v31; // r8
  int v32; // ecx
  unsigned int v33; // edx
  __int64 v34; // rsi
  int v35; // r11d
  __int64 *v36; // r10
  int v37; // ecx
  int v38; // ecx
  int v39; // r11d
  __int64 v40; // [rsp+8h] [rbp-38h]
  __int64 v41; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = a2 + 16;
    v11 = *a3;
    v12 = 96;
    v13 = 3;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 9) ^ (unsigned __int8)((unsigned int)*a3 >> 4)) & 3;
    v15 = (__int64 *)(v10 + 24LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
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
    v10 = *(_QWORD *)(a2 + 16);
    v11 = *a3;
    v13 = v17 - 1;
    v14 = (v17 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v15 = (__int64 *)(v10 + 24LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
      goto LABEL_6;
  }
  v26 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v16 == -4096 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      if ( !v19 )
        v19 = v15;
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      if ( v9 )
      {
        v21 = 12;
        LODWORD(v17) = 4;
LABEL_9:
        if ( v21 > 4 * v20 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v20 > (unsigned int)v17 >> 3 )
          {
LABEL_11:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a2 + 12);
            *v19 = *a3;
            v22 = *(_DWORD *)(a4 + 8);
            *(_DWORD *)(a4 + 8) = 0;
            *((_DWORD *)v19 + 4) = v22;
            v19[1] = *(_QWORD *)a4;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v23 = a2 + 16;
              v24 = 96;
            }
            else
            {
              v23 = *(_QWORD *)(a2 + 16);
              v24 = 24LL * *(unsigned int *)(a2 + 24);
            }
            v25 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v25;
            *(_QWORD *)(result + 24) = v24 + v23;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v41 = result;
          sub_2B56020(a2, v17);
          result = v41;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v31 = a2 + 16;
            v32 = 3;
            goto LABEL_29;
          }
          v38 = *(_DWORD *)(a2 + 24);
          v31 = *(_QWORD *)(a2 + 16);
          if ( v38 )
          {
            v32 = v38 - 1;
LABEL_29:
            v33 = v32 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v19 = (__int64 *)(v31 + 24LL * v33);
            v34 = *v19;
            if ( *v19 != *a3 )
            {
              v35 = 1;
              v36 = 0;
              while ( v34 != -4096 )
              {
                if ( v34 == -8192 && !v36 )
                  v36 = v19;
                v33 = v32 & (v35 + v33);
                v19 = (__int64 *)(v31 + 24LL * v33);
                v34 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_26;
                ++v35;
              }
LABEL_32:
              if ( v36 )
                v19 = v36;
              goto LABEL_26;
            }
            goto LABEL_26;
          }
LABEL_59:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v40 = result;
        sub_2B56020(a2, 2 * v17);
        result = v40;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v27 = a2 + 16;
          v28 = 3;
        }
        else
        {
          v37 = *(_DWORD *)(a2 + 24);
          v27 = *(_QWORD *)(a2 + 16);
          if ( !v37 )
            goto LABEL_59;
          v28 = v37 - 1;
        }
        v29 = v28 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (__int64 *)(v27 + 24LL * v29);
        v30 = *v19;
        if ( *a3 != *v19 )
        {
          v39 = 1;
          v36 = 0;
          while ( v30 != -4096 )
          {
            if ( !v36 && v30 == -8192 )
              v36 = v19;
            v29 = v28 & (v39 + v29);
            v19 = (__int64 *)(v27 + 24LL * v29);
            v30 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_26;
            ++v39;
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
    if ( v16 == -8192 && !v19 )
      v19 = v15;
    v14 = v13 & (v26 + v14);
    v15 = (__int64 *)(v10 + 24LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
      break;
    ++v26;
  }
  if ( v9 )
  {
    v12 = 96;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v12 = 24 * v17;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v8;
  *(_QWORD *)(result + 16) = v15;
  *(_QWORD *)(result + 24) = v10 + v12;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
