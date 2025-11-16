// Function: sub_307D460
// Address: 0x307d460
//
__int64 __fastcall sub_307D460(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r11
  __int64 v9; // r15
  __int64 v10; // r12
  int v11; // r14d
  unsigned int v12; // esi
  int *v13; // rcx
  int v14; // edi
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned int v18; // esi
  unsigned int v19; // edi
  int v20; // ecx
  _DWORD *v21; // rax
  __int64 v22; // rsi
  int v23; // ecx
  __int64 v24; // r8
  int v25; // esi
  unsigned int v26; // ecx
  int v27; // edi
  int *v28; // rax
  int v29; // ecx
  int v30; // ecx
  int v31; // ecx
  __int64 v32; // r8
  unsigned int v33; // esi
  int *v34; // r10
  int v35; // edi
  int v36; // eax
  int v37; // eax
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // rdi
  unsigned int v41; // r8d
  int v42; // esi
  int *v43; // r11
  int v44; // r10d
  int v45; // r8d
  __int64 v46; // [rsp-68h] [rbp-68h]
  __int64 v47; // [rsp-60h] [rbp-60h]
  __int64 v48; // [rsp-60h] [rbp-60h]
  __int64 v49; // [rsp-60h] [rbp-60h]
  int v50; // [rsp-60h] [rbp-60h]
  int v51; // [rsp-60h] [rbp-60h]
  unsigned int v52; // [rsp-58h] [rbp-58h]
  __int64 v53; // [rsp-58h] [rbp-58h]
  __int64 v54; // [rsp-58h] [rbp-58h]
  __int64 v55; // [rsp-58h] [rbp-58h]
  __int64 v56; // [rsp-58h] [rbp-58h]
  int v57; // [rsp-58h] [rbp-58h]
  int v58; // [rsp-58h] [rbp-58h]
  __int64 v59; // [rsp-50h] [rbp-50h]
  int v60[15]; // [rsp-3Ch] [rbp-3Ch] BYREF

  result = (unsigned int)*(unsigned __int16 *)(a2 + 68) - 14;
  if ( (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) > 4u )
  {
    result = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != 0 )
    {
      v6 = 0;
      v7 = a1;
      v8 = 40LL * (unsigned int)result;
      result = a1 + 56;
      v59 = a1 + 56;
      v9 = v8;
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = v6 + *(_QWORD *)(a2 + 32);
          if ( *(_BYTE *)v10 )
            goto LABEL_9;
          v11 = *(_DWORD *)(v10 + 8);
          if ( v11 >= 0 )
            goto LABEL_9;
          if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
            break;
LABEL_7:
          result = *(unsigned __int16 *)(a2 + 68);
          if ( !*(_WORD *)(a2 + 68) )
            goto LABEL_9;
          if ( (_DWORD)result == 68 )
            goto LABEL_9;
          v23 = *(_DWORD *)(v7 + 80);
          result = *(unsigned int *)(v10 + 8);
          v24 = *(_QWORD *)(v7 + 64);
          if ( !v23 )
            goto LABEL_9;
          v25 = v23 - 1;
          v26 = (v23 - 1) & (37 * result);
          v27 = *(_DWORD *)(v24 + 8LL * v26);
          if ( (_DWORD)result != v27 )
          {
            v44 = 1;
            while ( v27 != -1 )
            {
              v26 = v25 & (v44 + v26);
              v27 = *(_DWORD *)(v24 + 8LL * v26);
              if ( (_DWORD)result == v27 )
                goto LABEL_25;
              ++v44;
            }
            goto LABEL_9;
          }
LABEL_25:
          v48 = a3;
          v6 += 40;
          v54 = v7;
          v60[0] = *(_DWORD *)(v10 + 8);
          v28 = sub_307C080(v59, v60);
          a3 = v48;
          v7 = v54;
          v29 = *v28;
          result = (unsigned int)*v28 >> 6;
          *(_QWORD *)(*(_QWORD *)(v48 + 24) + 8 * result) |= 1LL << v29;
          if ( v9 == v6 )
            return result;
        }
        result = *(_QWORD *)(v7 + 64);
        v12 = *(_DWORD *)(v7 + 80);
        if ( !*(_WORD *)(a2 + 68) || *(_WORD *)(a2 + 68) == 68 )
          break;
        if ( v12 )
        {
          v18 = v12 - 1;
          v19 = v18 & (37 * v11);
          v20 = *(_DWORD *)(result + 8LL * v19);
          if ( v11 != v20 )
          {
            v45 = 1;
            while ( v20 != -1 )
            {
              v19 = v18 & (v45 + v19);
              v20 = *(_DWORD *)(result + 8LL * v19);
              if ( v11 == v20 )
                goto LABEL_22;
              ++v45;
            }
            goto LABEL_9;
          }
LABEL_22:
          v47 = a3;
          v53 = v7;
          v60[0] = *(_DWORD *)(v10 + 8);
          v21 = sub_307C080(v59, v60);
          a3 = v47;
          v7 = v53;
          v22 = *v21 >> 6;
          result = ~(1LL << *v21);
          *(_QWORD *)(*(_QWORD *)(v47 + 24) + 8 * v22) &= result;
LABEL_17:
          if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
            goto LABEL_7;
          v6 += 40;
          if ( v9 == v6 )
            return result;
        }
        else
        {
LABEL_9:
          v6 += 40;
          if ( v9 == v6 )
            return result;
        }
      }
      if ( v12 )
      {
        v52 = (v12 - 1) & (37 * v11);
        v13 = (int *)(result + 8LL * v52);
        v14 = *v13;
        if ( v11 == *v13 )
        {
LABEL_15:
          v15 = v13[1];
          v16 = 1LL << v15;
          v17 = 8LL * (v15 >> 6);
LABEL_16:
          result = *(_QWORD *)(a3 + 24) + v17;
          *(_QWORD *)result |= v16;
          goto LABEL_17;
        }
        v50 = 1;
        v34 = 0;
        while ( v14 != -1 )
        {
          if ( !v34 && v14 == -2 )
            v34 = v13;
          v52 = (v12 - 1) & (v50 + v52);
          v13 = (int *)(result + 8LL * v52);
          v14 = *v13;
          if ( v11 == *v13 )
            goto LABEL_15;
          ++v50;
        }
        v37 = *(_DWORD *)(v7 + 72);
        if ( !v34 )
          v34 = v13;
        ++*(_QWORD *)(v7 + 56);
        v36 = v37 + 1;
        if ( 4 * v36 < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(v7 + 76) - v36 > v12 >> 3 )
            goto LABEL_30;
          v56 = v7;
          v46 = a3;
          v51 = 37 * v11;
          sub_2E518D0(v59, v12);
          v7 = v56;
          v38 = *(_DWORD *)(v56 + 80);
          if ( !v38 )
          {
LABEL_71:
            ++*(_DWORD *)(v7 + 72);
            BUG();
          }
          v39 = v38 - 1;
          v40 = *(_QWORD *)(v56 + 64);
          a3 = v46;
          v41 = v39 & v51;
          v34 = (int *)(v40 + 8LL * (v39 & (unsigned int)v51));
          v42 = *v34;
          v36 = *(_DWORD *)(v56 + 72) + 1;
          if ( v11 == *v34 )
            goto LABEL_30;
          v57 = 1;
          v43 = 0;
          while ( v42 != -1 )
          {
            if ( v42 == -2 && !v43 )
              v43 = v34;
            v41 = v39 & (v57 + v41);
            v34 = (int *)(v40 + 8LL * v41);
            v42 = *v34;
            if ( v11 == *v34 )
              goto LABEL_30;
            ++v57;
          }
          goto LABEL_43;
        }
      }
      else
      {
        ++*(_QWORD *)(v7 + 56);
      }
      v55 = v7;
      v49 = a3;
      sub_2E518D0(v59, 2 * v12);
      v7 = v55;
      v30 = *(_DWORD *)(v55 + 80);
      if ( !v30 )
        goto LABEL_71;
      v31 = v30 - 1;
      v32 = *(_QWORD *)(v55 + 64);
      a3 = v49;
      v33 = v31 & (37 * v11);
      v34 = (int *)(v32 + 8LL * v33);
      v35 = *v34;
      v36 = *(_DWORD *)(v55 + 72) + 1;
      if ( v11 == *v34 )
        goto LABEL_30;
      v58 = 1;
      v43 = 0;
      while ( v35 != -1 )
      {
        if ( !v43 && v35 == -2 )
          v43 = v34;
        v33 = v31 & (v58 + v33);
        v34 = (int *)(v32 + 8LL * v33);
        v35 = *v34;
        if ( v11 == *v34 )
          goto LABEL_30;
        ++v58;
      }
LABEL_43:
      if ( v43 )
        v34 = v43;
LABEL_30:
      *(_DWORD *)(v7 + 72) = v36;
      if ( *v34 != -1 )
        --*(_DWORD *)(v7 + 76);
      *v34 = v11;
      v16 = 1;
      v17 = 0;
      v34[1] = 0;
      goto LABEL_16;
    }
  }
  return result;
}
