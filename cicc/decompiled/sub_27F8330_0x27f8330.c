// Function: sub_27F8330
// Address: 0x27f8330
//
__int64 __fastcall sub_27F8330(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 result; // rax
  _QWORD *v7; // rdi
  __int64 *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r13
  unsigned int v12; // esi
  __int64 v13; // r9
  unsigned int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // rcx
  int v17; // r11d
  __int64 *v18; // r8
  int v19; // eax
  __int64 *v20; // r13
  __int64 *v21; // r14
  __int64 v22; // r8
  __int64 *v23; // rcx
  __int64 v24; // rdi
  unsigned int v25; // esi
  int v26; // eax
  int v27; // r11d
  __int64 v28; // r9
  unsigned int v29; // edx
  __int64 *v30; // r10
  __int64 v31; // rdi
  int v32; // eax
  int v33; // esi
  __int64 *v34; // rcx
  int v35; // r11d
  int v36; // eax
  int v37; // eax
  int v38; // r11d
  __int64 v39; // r9
  int v40; // esi
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r9d
  __int64 v44; // r10
  unsigned int v45; // edx
  __int64 v46; // rdi
  int v47; // esi
  __int64 *v48; // rcx
  int v49; // edi
  int v50; // edi
  int v51; // edx
  __int64 *v52; // rsi
  unsigned int v53; // r14d
  __int64 v54; // rcx
  int v55; // [rsp+8h] [rbp-58h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  const void *v57; // [rsp+18h] [rbp-48h]
  __int64 v58[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a2;
  v57 = (const void *)(a1 + 48);
  result = a1 + 32;
  v56 = a1 + 32;
  if ( a2 != a3 )
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(*(_QWORD *)(v5 + 24) + 40LL);
      result = *(unsigned int *)(a1 + 16);
      v58[0] = v11;
      if ( !(_DWORD)result )
      {
        v7 = *(_QWORD **)(a1 + 32);
        v8 = &v7[*(unsigned int *)(a1 + 40)];
        result = (__int64)sub_27EBE50(v7, (__int64)v8, v58);
        if ( v8 != (__int64 *)result )
          goto LABEL_4;
        if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(v56, v57, v9 + 1, 8u, v9, v10);
          v8 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
        }
        *v8 = v11;
        result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
        *(_DWORD *)(a1 + 40) = result;
        if ( (unsigned int)result <= 8 )
          goto LABEL_4;
        v20 = *(__int64 **)(a1 + 32);
        v21 = &v20[result];
        while ( 2 )
        {
          v25 = *(_DWORD *)(a1 + 24);
          if ( !v25 )
          {
            ++*(_QWORD *)a1;
            goto LABEL_29;
          }
          v22 = *(_QWORD *)(a1 + 8);
          result = (v25 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
          v23 = (__int64 *)(v22 + 8 * result);
          v24 = *v23;
          if ( *v20 == *v23 )
          {
LABEL_26:
            if ( v21 == ++v20 )
              goto LABEL_4;
            continue;
          }
          break;
        }
        v35 = 1;
        v30 = 0;
        while ( v24 != -4096 )
        {
          if ( v24 != -8192 || v30 )
            v23 = v30;
          result = (v25 - 1) & (v35 + (_DWORD)result);
          v24 = *(_QWORD *)(v22 + 8LL * (unsigned int)result);
          if ( *v20 == v24 )
            goto LABEL_26;
          ++v35;
          v30 = v23;
          v23 = (__int64 *)(v22 + 8LL * (unsigned int)result);
        }
        v36 = *(_DWORD *)(a1 + 16);
        if ( !v30 )
          v30 = v23;
        ++*(_QWORD *)a1;
        v32 = v36 + 1;
        if ( 4 * v32 >= 3 * v25 )
        {
LABEL_29:
          sub_CF28B0(a1, 2 * v25);
          v26 = *(_DWORD *)(a1 + 24);
          if ( !v26 )
            goto LABEL_93;
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 8);
          v29 = (v26 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
          v30 = (__int64 *)(v28 + 8LL * v29);
          v31 = *v30;
          v32 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v20 != *v30 )
          {
            v33 = 1;
            v34 = 0;
            while ( v31 != -4096 )
            {
              if ( v31 == -8192 && !v34 )
                v34 = v30;
              v29 = v27 & (v33 + v29);
              v30 = (__int64 *)(v28 + 8LL * v29);
              v31 = *v30;
              if ( *v20 == *v30 )
                goto LABEL_45;
              ++v33;
            }
            goto LABEL_51;
          }
        }
        else if ( v25 - *(_DWORD *)(a1 + 20) - v32 <= v25 >> 3 )
        {
          sub_CF28B0(a1, v25);
          v37 = *(_DWORD *)(a1 + 24);
          if ( !v37 )
            goto LABEL_93;
          v38 = v37 - 1;
          v39 = *(_QWORD *)(a1 + 8);
          v34 = 0;
          v40 = 1;
          v41 = (v37 - 1) & (((unsigned int)*v20 >> 9) ^ ((unsigned int)*v20 >> 4));
          v30 = (__int64 *)(v39 + 8LL * v41);
          v42 = *v30;
          v32 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v20 != *v30 )
          {
            while ( v42 != -4096 )
            {
              if ( !v34 && v42 == -8192 )
                v34 = v30;
              v41 = v38 & (v40 + v41);
              v30 = (__int64 *)(v39 + 8LL * v41);
              v42 = *v30;
              if ( *v20 == *v30 )
                goto LABEL_45;
              ++v40;
            }
LABEL_51:
            if ( v34 )
              v30 = v34;
          }
        }
LABEL_45:
        *(_DWORD *)(a1 + 16) = v32;
        if ( *v30 != -4096 )
          --*(_DWORD *)(a1 + 20);
        result = *v20;
        *v30 = *v20;
        goto LABEL_26;
      }
      v12 = *(_DWORD *)(a1 + 24);
      if ( !v12 )
        break;
      v13 = *(_QWORD *)(a1 + 8);
      v14 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v15 = (__int64 *)(v13 + 8LL * v14);
      v16 = *v15;
      if ( v11 != *v15 )
      {
        v55 = result;
        v17 = 1;
        v18 = 0;
        while ( v16 != -4096 )
        {
          if ( v18 || v16 != -8192 )
            v15 = v18;
          v14 = (v12 - 1) & (v17 + v14);
          result = v13 + 8LL * v14;
          v16 = *(_QWORD *)result;
          if ( v11 == *(_QWORD *)result )
            goto LABEL_4;
          ++v17;
          v18 = v15;
          v15 = (__int64 *)(v13 + 8LL * v14);
        }
        if ( !v18 )
          v18 = v15;
        ++*(_QWORD *)a1;
        v19 = v55 + 1;
        if ( 4 * (v55 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 20) - v19 <= v12 >> 3 )
          {
            sub_CF28B0(a1, v12);
            v49 = *(_DWORD *)(a1 + 24);
            if ( !v49 )
            {
LABEL_93:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v50 = v49 - 1;
            v13 = *(_QWORD *)(a1 + 8);
            v51 = 1;
            v52 = 0;
            v53 = v50 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v18 = (__int64 *)(v13 + 8LL * v53);
            v54 = *v18;
            v19 = *(_DWORD *)(a1 + 16) + 1;
            if ( v11 != *v18 )
            {
              while ( v54 != -4096 )
              {
                if ( !v52 && v54 == -8192 )
                  v52 = v18;
                v53 = v50 & (v51 + v53);
                v18 = (__int64 *)(v13 + 8LL * v53);
                v54 = *v18;
                if ( v11 == *v18 )
                  goto LABEL_16;
                ++v51;
              }
              if ( v52 )
                v18 = v52;
            }
          }
          goto LABEL_16;
        }
LABEL_55:
        sub_CF28B0(a1, 2 * v12);
        v43 = *(_DWORD *)(a1 + 24);
        if ( !v43 )
          goto LABEL_93;
        v13 = (unsigned int)(v43 - 1);
        v44 = *(_QWORD *)(a1 + 8);
        v45 = v13 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v18 = (__int64 *)(v44 + 8LL * v45);
        v46 = *v18;
        v19 = *(_DWORD *)(a1 + 16) + 1;
        if ( v11 != *v18 )
        {
          v47 = 1;
          v48 = 0;
          while ( v46 != -4096 )
          {
            if ( !v48 && v46 == -8192 )
              v48 = v18;
            v45 = v13 & (v47 + v45);
            v18 = (__int64 *)(v44 + 8LL * v45);
            v46 = *v18;
            if ( v11 == *v18 )
              goto LABEL_16;
            ++v47;
          }
          if ( v48 )
            v18 = v48;
        }
LABEL_16:
        *(_DWORD *)(a1 + 16) = v19;
        if ( *v18 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v18 = v11;
        result = *(unsigned int *)(a1 + 40);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(v56, v57, result + 1, 8u, (__int64)v18, v13);
          result = *(unsigned int *)(a1 + 40);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * result) = v11;
        ++*(_DWORD *)(a1 + 40);
        goto LABEL_4;
      }
      do
      {
LABEL_4:
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          break;
        result = (unsigned int)**(unsigned __int8 **)(v5 + 24) - 30;
      }
      while ( (unsigned __int8)(**(_BYTE **)(v5 + 24) - 30) > 0xAu );
      if ( a3 == v5 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_55;
  }
  return result;
}
