// Function: sub_1C0BEB0
// Address: 0x1c0beb0
//
__int64 __fastcall sub_1C0BEB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 result; // rax
  unsigned int v9; // esi
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // r8
  __int64 *v15; // rbx
  __int64 *v16; // r12
  __int64 v17; // r8
  __int64 v18; // rax
  int v19; // edx
  int v20; // ecx
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 v23; // rsi
  int v24; // r9d
  int v25; // edx
  _QWORD *v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // r10
  __int64 v29; // rsi
  _QWORD *v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // r9
  unsigned int v33; // edx
  __int64 *v34; // rdi
  __int64 v35; // r8
  int v36; // r11d
  __int64 *v37; // rcx
  int v38; // edi
  int v39; // edx
  int v40; // ebx
  __int64 *v41; // rcx
  int v42; // edi
  int v43; // edx
  int v44; // edx
  int v45; // r11d
  __int64 v46; // r10
  unsigned int v47; // esi
  int v48; // edx
  __int64 v49; // r9
  int v50; // r8d
  __int64 *v51; // rdi
  __int64 v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  __int64 v55; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v56[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a2 + 32);
  result = *(_QWORD *)(a2 + 72);
  v54 = 0;
  v53 = result;
  v52 = 8 * v5;
  if ( (_DWORD)v5 )
  {
LABEL_2:
    while ( 1 )
    {
      v9 = *(_DWORD *)(a4 + 24);
      v10 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + v54);
      v55 = v10;
      if ( !v9 )
        break;
      v11 = *(_QWORD *)(a4 + 8);
      LODWORD(v12) = (v9 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v13 = (__int64 *)(v11 + 40LL * (unsigned int)v12);
      v14 = *v13;
      if ( v10 == *v13 )
      {
LABEL_4:
        v15 = (__int64 *)v13[2];
        v16 = &v15[*((unsigned int *)v13 + 8)];
        if ( !*((_DWORD *)v13 + 6) || v16 == v15 )
          goto LABEL_9;
        while ( *v15 == -16 || *v15 == -8 )
        {
          if ( ++v15 == v16 )
            goto LABEL_9;
        }
        if ( v16 == v15 )
          goto LABEL_9;
        while ( 2 )
        {
          v17 = *v15;
          if ( a2 == *v15 )
            goto LABEL_15;
          v18 = *(_QWORD *)v17;
          v19 = *(_DWORD *)(a3 + 24);
          v55 = *(_QWORD *)v17;
          if ( v19 )
          {
            v20 = v19 - 1;
            v21 = *(_QWORD *)(a3 + 8);
            v22 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v23 = *(_QWORD *)(v21 + 8LL * v22);
            if ( v18 == v23 )
              goto LABEL_15;
            v24 = 1;
            while ( v23 != -8 )
            {
              v22 = v20 & (v24 + v22);
              v23 = *(_QWORD *)(v21 + 8LL * v22);
              if ( v18 == v23 )
                goto LABEL_15;
              ++v24;
            }
          }
          v25 = *(_DWORD *)(v17 + 32);
          if ( v25 == *(_DWORD *)(a2 + 32) )
          {
            if ( v25 )
            {
              v26 = *(_QWORD **)(v17 + 24);
              v27 = (unsigned int)(v25 - 1);
              v28 = (__int64)&v26[v27 + 1];
              v29 = v27 * 8 + *(_QWORD *)(a2 + 24) + 8;
              do
              {
                v30 = *(_QWORD **)(a2 + 24);
                while ( *v30 != *v26 )
                {
                  if ( (_QWORD *)v29 == ++v30 )
                    goto LABEL_15;
                }
                ++v26;
              }
              while ( (_QWORD *)v28 != v26 );
            }
            *(_QWORD *)(v17 + 72) = v53;
            v31 = *(_DWORD *)(a3 + 24);
            if ( !v31 )
            {
              ++*(_QWORD *)a3;
              goto LABEL_59;
            }
            v32 = *(_QWORD *)(a3 + 8);
            v33 = (v31 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v34 = (__int64 *)(v32 + 8LL * v33);
            v35 = *v34;
            if ( v18 != *v34 )
            {
              v36 = 1;
              v37 = 0;
              while ( v35 != -8 )
              {
                if ( v35 == -16 && !v37 )
                  v37 = v34;
                v33 = (v31 - 1) & (v36 + v33);
                v34 = (__int64 *)(v32 + 8LL * v33);
                v35 = *v34;
                if ( v18 == *v34 )
                  goto LABEL_15;
                ++v36;
              }
              if ( !v37 )
                v37 = v34;
              v38 = *(_DWORD *)(a3 + 16);
              ++*(_QWORD *)a3;
              v39 = v38 + 1;
              if ( 4 * (v38 + 1) < 3 * v31 )
              {
                if ( v31 - *(_DWORD *)(a3 + 20) - v39 <= v31 >> 3 )
                {
                  sub_13B3D40(a3, v31);
                  sub_1898220(a3, &v55, v56);
                  v37 = (__int64 *)v56[0];
                  v18 = v55;
                  v39 = *(_DWORD *)(a3 + 16) + 1;
                }
                goto LABEL_41;
              }
LABEL_59:
              sub_13B3D40(a3, 2 * v31);
              v44 = *(_DWORD *)(a3 + 24);
              if ( !v44 )
              {
                ++*(_DWORD *)(a3 + 16);
                BUG();
              }
              v18 = v55;
              v45 = v44 - 1;
              v46 = *(_QWORD *)(a3 + 8);
              v47 = (v44 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
              v37 = (__int64 *)(v46 + 8LL * v47);
              v48 = *(_DWORD *)(a3 + 16);
              v49 = *v37;
              if ( *v37 == v55 )
              {
LABEL_61:
                v39 = v48 + 1;
              }
              else
              {
                v50 = 1;
                v51 = 0;
                while ( v49 != -8 )
                {
                  if ( !v51 && v49 == -16 )
                    v51 = v37;
                  v47 = v45 & (v50 + v47);
                  v37 = (__int64 *)(v46 + 8LL * v47);
                  v49 = *v37;
                  if ( v55 == *v37 )
                    goto LABEL_61;
                  ++v50;
                }
                v39 = v48 + 1;
                if ( v51 )
                  v37 = v51;
              }
LABEL_41:
              *(_DWORD *)(a3 + 16) = v39;
              if ( *v37 != -8 )
                --*(_DWORD *)(a3 + 20);
              *v37 = v18;
            }
          }
LABEL_15:
          if ( ++v15 != v16 )
          {
            while ( *v15 == -8 || *v15 == -16 )
            {
              if ( v16 == ++v15 )
              {
                v54 += 8;
                result = v54;
                if ( v52 != v54 )
                  goto LABEL_2;
                return result;
              }
            }
            if ( v15 != v16 )
              continue;
          }
          goto LABEL_9;
        }
      }
      v40 = 1;
      v41 = 0;
      while ( v14 != -8 )
      {
        if ( v14 == -16 && !v41 )
          v41 = v13;
        v12 = (v9 - 1) & ((_DWORD)v12 + v40);
        v13 = (__int64 *)(v11 + 40 * v12);
        v14 = *v13;
        if ( v10 == *v13 )
          goto LABEL_4;
        ++v40;
      }
      if ( !v41 )
        v41 = v13;
      v42 = *(_DWORD *)(a4 + 16);
      ++*(_QWORD *)a4;
      v43 = v42 + 1;
      if ( 4 * (v42 + 1) >= 3 * v9 )
        goto LABEL_56;
      if ( v9 - *(_DWORD *)(a4 + 20) - v43 <= v9 >> 3 )
        goto LABEL_57;
LABEL_50:
      *(_DWORD *)(a4 + 16) = v43;
      if ( *v41 != -8 )
        --*(_DWORD *)(a4 + 20);
      *v41 = v10;
      v41[1] = 0;
      v41[2] = 0;
      v41[3] = 0;
      *((_DWORD *)v41 + 8) = 0;
LABEL_9:
      v54 += 8;
      result = v54;
      if ( v52 == v54 )
        return result;
    }
    ++*(_QWORD *)a4;
LABEL_56:
    v9 *= 2;
LABEL_57:
    sub_1C0BC70(a4, v9);
    sub_1C09AC0(a4, &v55, v56);
    v41 = (__int64 *)v56[0];
    v10 = v55;
    v43 = *(_DWORD *)(a4 + 16) + 1;
    goto LABEL_50;
  }
  return result;
}
