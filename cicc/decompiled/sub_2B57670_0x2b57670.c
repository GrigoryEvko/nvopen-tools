// Function: sub_2B57670
// Address: 0x2b57670
//
__int64 __fastcall sub_2B57670(__int64 a1, __int64 a2, __int64 *a3, _QWORD *a4, unsigned int *a5)
{
  __int64 result; // rax
  __int64 v9; // r10
  char v10; // di
  __int64 v11; // r9
  __int64 v12; // r11
  __int64 v13; // rsi
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r15
  __int64 v17; // rsi
  unsigned int v18; // edx
  __int64 *v19; // r14
  int v20; // ecx
  unsigned int v21; // r9d
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  int v25; // esi
  __int64 v26; // rsi
  int v27; // ecx
  unsigned int v28; // edx
  __int64 v29; // r9
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 v33; // r9
  int v34; // r11d
  __int64 *v35; // r10
  int v36; // ecx
  int v37; // ecx
  int v38; // r11d
  unsigned int *v39; // [rsp+0h] [rbp-40h]
  unsigned int *v40; // [rsp+0h] [rbp-40h]
  int v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]

  result = a1;
  v9 = *(_QWORD *)a2;
  v10 = *(_BYTE *)(a2 + 8) & 1;
  if ( v10 )
  {
    v11 = *a3;
    v12 = a2 + 16;
    v13 = 64;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 3;
    v15 = (__int64 *)(v12
                    + 16LL
                    * (((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 3));
    v16 = *v15;
    if ( v11 == *v15 )
      goto LABEL_3;
    v41 = 3;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v17 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v19 = 0;
      *(_QWORD *)a2 = v9 + 1;
      v20 = (v18 >> 1) + 1;
      goto LABEL_8;
    }
    v11 = *a3;
    v12 = *(_QWORD *)(a2 + 16);
    v41 = v17 - 1;
    v14 = (v17 - 1) & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
      goto LABEL_6;
  }
  v25 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v16 == -4096 )
    {
      if ( !v19 )
        v19 = v15;
      v18 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v9 + 1;
      v20 = (v18 >> 1) + 1;
      if ( v10 )
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
            *v19 = *a3;
            v19[1] = (4LL * *a5) | *a4 & 0xFFFFFFFFFFFFFFFBLL;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v22 = a2 + 16;
              v23 = 64;
            }
            else
            {
              v22 = *(_QWORD *)(a2 + 16);
              v23 = 16LL * *(unsigned int *)(a2 + 24);
            }
            v24 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v24;
            *(_QWORD *)(result + 24) = v23 + v22;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v40 = a5;
          v43 = result;
          sub_2B57250(a2, v17);
          result = v43;
          a5 = v40;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v30 = a2 + 16;
            v31 = 3;
            goto LABEL_30;
          }
          v37 = *(_DWORD *)(a2 + 24);
          v30 = *(_QWORD *)(a2 + 16);
          if ( v37 )
          {
            v31 = v37 - 1;
LABEL_30:
            v32 = v31 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v19 = (__int64 *)(v30 + 16LL * v32);
            v33 = *v19;
            if ( *v19 != *a3 )
            {
              v34 = 1;
              v35 = 0;
              while ( v33 != -4096 )
              {
                if ( v33 == -8192 && !v35 )
                  v35 = v19;
                v32 = v31 & (v34 + v32);
                v19 = (__int64 *)(v30 + 16LL * v32);
                v33 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_27;
                ++v34;
              }
LABEL_33:
              if ( v35 )
                v19 = v35;
              goto LABEL_27;
            }
            goto LABEL_27;
          }
LABEL_60:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v39 = a5;
        v42 = result;
        sub_2B57250(a2, 2 * v17);
        result = v42;
        a5 = v39;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v26 = a2 + 16;
          v27 = 3;
        }
        else
        {
          v36 = *(_DWORD *)(a2 + 24);
          v26 = *(_QWORD *)(a2 + 16);
          if ( !v36 )
            goto LABEL_60;
          v27 = v36 - 1;
        }
        v28 = v27 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v19 = (__int64 *)(v26 + 16LL * v28);
        v29 = *v19;
        if ( *a3 != *v19 )
        {
          v38 = 1;
          v35 = 0;
          while ( v29 != -4096 )
          {
            if ( !v35 && v29 == -8192 )
              v35 = v19;
            v28 = v27 & (v38 + v28);
            v19 = (__int64 *)(v26 + 16LL * v28);
            v29 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_27;
            ++v38;
          }
          goto LABEL_33;
        }
LABEL_27:
        v18 = *(_DWORD *)(a2 + 8);
        goto LABEL_11;
      }
      LODWORD(v17) = *(_DWORD *)(a2 + 24);
LABEL_8:
      v21 = 3 * v17;
      goto LABEL_9;
    }
    if ( !v19 && v16 == -8192 )
      v19 = v15;
    v14 = v41 & (v25 + v14);
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( *v15 == v11 )
      break;
    ++v25;
  }
  if ( v10 )
  {
    v13 = 64;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v13 = 16 * v17;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v9;
  *(_QWORD *)(result + 16) = v15;
  *(_QWORD *)(result + 24) = v12 + v13;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
