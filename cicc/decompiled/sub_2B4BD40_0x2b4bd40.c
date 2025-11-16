// Function: sub_2B4BD40
// Address: 0x2b4bd40
//
__int64 __fastcall sub_2B4BD40(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // r10
  char v7; // di
  __int64 v8; // r8
  __int64 v9; // r11
  unsigned int v10; // edx
  _QWORD *v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rsi
  int v15; // ecx
  unsigned __int64 v16; // rdx
  _QWORD *v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned int v20; // edi
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // rsi
  int v26; // esi
  __int64 v27; // r8
  int v28; // ecx
  unsigned int v29; // edx
  __int64 v30; // rsi
  __int64 v31; // r8
  int v32; // esi
  unsigned int v33; // edx
  __int64 v34; // rcx
  int v35; // r11d
  _QWORD *v36; // r10
  int v37; // ecx
  int v38; // esi
  int v39; // r11d
  __int64 v40; // [rsp+8h] [rbp-38h]
  __int64 v41; // [rsp+8h] [rbp-38h]

  result = a1;
  v6 = *(_QWORD *)a2;
  v7 = *(_BYTE *)(a2 + 8) & 1;
  if ( v7 )
  {
    v8 = *a3;
    v9 = a2 + 16;
    v10 = ((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 3;
    v11 = (_QWORD *)(a2 + 16 + 72LL * v10);
    v12 = 288;
    v13 = *v11;
    if ( v8 == *v11 )
      goto LABEL_3;
    v15 = 3;
  }
  else
  {
    v14 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v14 )
    {
      v16 = *(unsigned int *)(a2 + 8);
      v17 = 0;
      *(_QWORD *)a2 = v6 + 1;
      v18 = ((unsigned int)v16 >> 1) + 1;
      goto LABEL_8;
    }
    v8 = *a3;
    v15 = v14 - 1;
    v9 = *(_QWORD *)(a2 + 16);
    v10 = (v14 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v11 = (_QWORD *)(v9 + 72LL * v10);
    v13 = *v11;
    if ( v8 == *v11 )
      goto LABEL_6;
  }
  v26 = 1;
  v17 = 0;
  while ( 1 )
  {
    if ( v13 == -4096 )
    {
      v16 = *(unsigned int *)(a2 + 8);
      if ( !v17 )
        v17 = v11;
      *(_QWORD *)a2 = v6 + 1;
      v18 = ((unsigned int)v16 >> 1) + 1;
      if ( v7 )
      {
        v19 = 12;
        LODWORD(v14) = 4;
LABEL_9:
        if ( 4 * (int)v18 < (unsigned int)v19 )
        {
          v20 = v14 - *(_DWORD *)(a2 + 12) - v18;
          v21 = (unsigned int)v14 >> 3;
          if ( v20 > (unsigned int)v21 )
          {
LABEL_11:
            *(_DWORD *)(a2 + 8) = (2 * ((unsigned int)v16 >> 1) + 2) | v16 & 1;
            if ( *v17 != -4096 )
              --*(_DWORD *)(a2 + 12);
            v22 = *a3;
            v17[2] = 0x600000000LL;
            *v17 = v22;
            v17[1] = v17 + 3;
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v23 = a2 + 16;
              v24 = 288;
            }
            else
            {
              v23 = *(_QWORD *)(a2 + 16);
              v24 = 72LL * *(unsigned int *)(a2 + 24);
            }
            v25 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v17;
            *(_QWORD *)(result + 8) = v25;
            *(_QWORD *)(result + 24) = v24 + v23;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v41 = result;
          sub_2B4B7F0(a2, v14, v16, v21, v19, (__int64)v17);
          result = v41;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v31 = a2 + 16;
            v32 = 3;
            goto LABEL_30;
          }
          v38 = *(_DWORD *)(a2 + 24);
          v31 = *(_QWORD *)(a2 + 16);
          if ( v38 )
          {
            v32 = v38 - 1;
LABEL_30:
            v33 = v32 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
            v17 = (_QWORD *)(v31 + 72LL * v33);
            v34 = *v17;
            if ( *v17 != *a3 )
            {
              v35 = 1;
              v36 = 0;
              while ( v34 != -4096 )
              {
                if ( v34 == -8192 && !v36 )
                  v36 = v17;
                v33 = v32 & (v35 + v33);
                v17 = (_QWORD *)(v31 + 72LL * v33);
                v34 = *v17;
                if ( *a3 == *v17 )
                  goto LABEL_27;
                ++v35;
              }
LABEL_33:
              if ( v36 )
                v17 = v36;
              goto LABEL_27;
            }
            goto LABEL_27;
          }
LABEL_60:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v40 = result;
        sub_2B4B7F0(a2, 2 * v14, v16, v18, v19, (__int64)v17);
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
            goto LABEL_60;
          v28 = v37 - 1;
        }
        v29 = v28 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
        v17 = (_QWORD *)(v27 + 72LL * v29);
        v30 = *v17;
        if ( *a3 != *v17 )
        {
          v39 = 1;
          v36 = 0;
          while ( v30 != -4096 )
          {
            if ( !v36 && v30 == -8192 )
              v36 = v17;
            v29 = v28 & (v39 + v29);
            v17 = (_QWORD *)(v27 + 72LL * v29);
            v30 = *v17;
            if ( *a3 == *v17 )
              goto LABEL_27;
            ++v39;
          }
          goto LABEL_33;
        }
LABEL_27:
        LODWORD(v16) = *(_DWORD *)(a2 + 8);
        goto LABEL_11;
      }
      LODWORD(v14) = *(_DWORD *)(a2 + 24);
LABEL_8:
      v19 = (unsigned int)(3 * v14);
      goto LABEL_9;
    }
    if ( !v17 && v13 == -8192 )
      v17 = v11;
    v10 = v15 & (v26 + v10);
    v11 = (_QWORD *)(v9 + 72LL * v10);
    v13 = *v11;
    if ( *v11 == v8 )
      break;
    ++v26;
  }
  if ( v7 )
  {
    v12 = 288;
  }
  else
  {
    v14 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v12 = 72 * v14;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v6;
  *(_QWORD *)(result + 16) = v11;
  *(_QWORD *)(result + 24) = v9 + v12;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
