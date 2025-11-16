// Function: sub_2E52320
// Address: 0x2e52320
//
__int64 __fastcall sub_2E52320(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v9; // rbx
  __int64 v10; // r14
  int v11; // edx
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // edx
  int v15; // edi
  __int64 v16; // r13
  int v17; // edx
  __int64 v18; // rdi
  int v19; // edx
  unsigned int v20; // r8d
  int v21; // esi
  unsigned int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // ecx
  _DWORD *v25; // rdx
  int v26; // r8d
  _QWORD *v27; // rdx
  int v28; // r8d
  unsigned int v29; // esi
  __int64 v30; // r8
  unsigned int v31; // r9d
  _DWORD *v32; // rdx
  int v33; // ecx
  int v34; // r9d
  _DWORD *v35; // rdi
  int v36; // ecx
  int v37; // ecx
  int v38; // edx
  int v39; // edx
  __int64 v40; // r9
  unsigned int v41; // ecx
  int v42; // esi
  int v43; // ecx
  int v44; // ecx
  int v45; // ecx
  int v46; // r13d
  _DWORD *v47; // r8
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+18h] [rbp-58h]
  int v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  int v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  _DWORD *v58; // [rsp+20h] [rbp-50h]
  _DWORD *v59; // [rsp+20h] [rbp-50h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  unsigned int v61; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v62[7]; // [rsp+38h] [rbp-38h] BYREF

  result = a2 + 48;
  v5 = *(_QWORD *)(a2 + 56);
  v60 = a2 + 48;
  if ( v5 != a2 + 48 )
  {
    while ( 1 )
    {
      result = *(_DWORD *)(v5 + 40) & 0xFFFFFF;
      if ( (*(_DWORD *)(v5 + 40) & 0xFFFFFF) != 0 )
        break;
LABEL_16:
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( v60 == v5 )
        return result;
    }
    result = (unsigned int)result;
    v9 = 0;
    v10 = 40LL * (unsigned int)result;
    while ( 1 )
    {
      while ( 1 )
      {
        v16 = v9 + *(_QWORD *)(v5 + 32);
        if ( *(_BYTE *)v16 )
          goto LABEL_6;
        result = *(unsigned int *)(v16 + 8);
        if ( (int)result >= 0 )
          goto LABEL_6;
        if ( (*(_BYTE *)(v16 + 3) & 0x10) == 0 )
          goto LABEL_10;
        v11 = *(_DWORD *)(a1 + 80);
        v12 = *(_QWORD *)(a1 + 64);
        if ( v11 )
          break;
LABEL_24:
        v29 = *(_DWORD *)(a4 + 24);
        v61 = *(_DWORD *)(v16 + 8);
        if ( !v29 )
        {
          ++*(_QWORD *)a4;
          v62[0] = 0;
          goto LABEL_60;
        }
        v30 = *(_QWORD *)(a4 + 8);
        v31 = (v29 - 1) & (37 * result);
        v32 = (_DWORD *)(v30 + 4LL * v31);
        v33 = *v32;
        if ( (_DWORD)result != *v32 )
        {
          v52 = 1;
          v58 = 0;
          while ( v33 != -1 )
          {
            if ( v58 || v33 != -2 )
              v32 = v58;
            v31 = (v29 - 1) & (v52 + v31);
            v33 = *(_DWORD *)(v30 + 4LL * v31);
            if ( (_DWORD)result == v33 )
              goto LABEL_26;
            v58 = v32;
            v32 = (_DWORD *)(v30 + 4LL * v31);
            ++v52;
          }
          if ( v58 )
            v32 = v58;
          v44 = *(_DWORD *)(a4 + 16) + 1;
          ++*(_QWORD *)a4;
          v59 = v32;
          v62[0] = v32;
          if ( 4 * v44 < 3 * v29 )
          {
            if ( v29 - *(_DWORD *)(a4 + 20) - v44 <= v29 >> 3 )
            {
              v49 = a3;
              v54 = a1;
              sub_A08C50(a4, v29);
              sub_22B31A0(a4, (int *)&v61, v62);
              result = v61;
              a3 = v49;
              a1 = v54;
              v44 = *(_DWORD *)(a4 + 16) + 1;
              v59 = (_DWORD *)v62[0];
            }
            goto LABEL_53;
          }
LABEL_60:
          v48 = a3;
          v53 = a1;
          sub_A08C50(a4, 2 * v29);
          sub_22B31A0(a4, (int *)&v61, v62);
          result = v61;
          a1 = v53;
          a3 = v48;
          v44 = *(_DWORD *)(a4 + 16) + 1;
          v59 = (_DWORD *)v62[0];
LABEL_53:
          *(_DWORD *)(a4 + 16) = v44;
          if ( *v59 != -1 )
            --*(_DWORD *)(a4 + 20);
          *v59 = result;
        }
LABEL_26:
        if ( (*(_BYTE *)(v16 + 3) & 0x10) != 0 )
          goto LABEL_6;
        result = *(unsigned int *)(v16 + 8);
LABEL_10:
        v17 = *(_DWORD *)(a4 + 24);
        v18 = *(_QWORD *)(a4 + 8);
        if ( !v17 )
          goto LABEL_6;
        v19 = v17 - 1;
        v20 = v19 & (37 * result);
        v21 = *(_DWORD *)(v18 + 4LL * v20);
        if ( v21 != (_DWORD)result )
        {
          v34 = 1;
          while ( v21 != -1 )
          {
            v20 = v19 & (v34 + v20);
            v21 = *(_DWORD *)(v18 + 4LL * v20);
            if ( v21 == (_DWORD)result )
              goto LABEL_12;
            ++v34;
          }
          goto LABEL_6;
        }
LABEL_12:
        v22 = *(_DWORD *)(a3 + 24);
        v61 = result;
        if ( !v22 )
        {
          ++*(_QWORD *)a3;
          v62[0] = 0;
          goto LABEL_43;
        }
        v23 = *(_QWORD *)(a3 + 8);
        v24 = (v22 - 1) & (37 * result);
        v25 = (_DWORD *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( *v25 != (_DWORD)result )
        {
          v55 = 1;
          v35 = 0;
          while ( v26 != -1 )
          {
            if ( !v35 && v26 == -2 )
              v35 = v25;
            v24 = (v22 - 1) & (v55 + v24);
            v25 = (_DWORD *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( *v25 == (_DWORD)result )
              goto LABEL_14;
            ++v55;
          }
          v36 = *(_DWORD *)(a3 + 16);
          if ( !v35 )
            v35 = v25;
          ++*(_QWORD *)a3;
          v37 = v36 + 1;
          v62[0] = v35;
          if ( 4 * v37 < 3 * v22 )
          {
            if ( v22 - *(_DWORD *)(a3 + 20) - v37 <= v22 >> 3 )
            {
              v51 = a1;
              v57 = a3;
              sub_2E51AA0(a3, v22);
              sub_2E50670(v57, (int *)&v61, v62);
              a3 = v57;
              result = v61;
              v35 = (_DWORD *)v62[0];
              a1 = v51;
              v37 = *(_DWORD *)(v57 + 16) + 1;
            }
LABEL_39:
            *(_DWORD *)(a3 + 16) = v37;
            if ( *v35 != -1 )
              --*(_DWORD *)(a3 + 20);
            *v35 = result;
            v27 = v35 + 2;
            *((_QWORD *)v35 + 1) = 0;
            goto LABEL_15;
          }
LABEL_43:
          v50 = a1;
          v56 = a3;
          sub_2E51AA0(a3, 2 * v22);
          a3 = v56;
          a1 = v50;
          v38 = *(_DWORD *)(v56 + 24);
          if ( v38 )
          {
            result = v61;
            v39 = v38 - 1;
            v40 = *(_QWORD *)(v56 + 8);
            v41 = v39 & (37 * v61);
            v35 = (_DWORD *)(v40 + 16LL * v41);
            v42 = *v35;
            if ( *v35 == v61 )
            {
LABEL_45:
              v43 = *(_DWORD *)(v56 + 16);
              v62[0] = v35;
              v37 = v43 + 1;
            }
            else
            {
              v46 = 1;
              v47 = 0;
              while ( v42 != -1 )
              {
                if ( v42 == -2 && !v47 )
                  v47 = v35;
                v41 = v39 & (v46 + v41);
                v35 = (_DWORD *)(v40 + 16LL * v41);
                v42 = *v35;
                if ( v61 == *v35 )
                  goto LABEL_45;
                ++v46;
              }
              if ( !v47 )
                v47 = v35;
              v37 = *(_DWORD *)(v56 + 16) + 1;
              v62[0] = v47;
              v35 = v47;
            }
          }
          else
          {
            v45 = *(_DWORD *)(v56 + 16);
            result = v61;
            v62[0] = 0;
            v35 = 0;
            v37 = v45 + 1;
          }
          goto LABEL_39;
        }
LABEL_14:
        v27 = v25 + 2;
LABEL_15:
        v9 += 40;
        *v27 = v5;
        if ( v10 == v9 )
          goto LABEL_16;
      }
      v13 = v11 - 1;
      v14 = (v11 - 1) & (37 * result);
      v15 = *(_DWORD *)(v12 + 8LL * v14);
      if ( (_DWORD)result != v15 )
      {
        v28 = 1;
        while ( v15 != -1 )
        {
          v14 = v13 & (v28 + v14);
          v15 = *(_DWORD *)(v12 + 8LL * v14);
          if ( (_DWORD)result == v15 )
            goto LABEL_6;
          ++v28;
        }
        goto LABEL_24;
      }
LABEL_6:
      v9 += 40;
      if ( v10 == v9 )
        goto LABEL_16;
    }
  }
  return result;
}
