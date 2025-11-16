// Function: sub_36FAF40
// Address: 0x36faf40
//
__int64 __fastcall sub_36FAF40(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax
  void *v7; // rdi
  __int64 *v8; // rbx
  __int64 *v9; // r12
  unsigned int v11; // ecx
  unsigned int v12; // edx
  _DWORD *v13; // rdi
  int v14; // ebx
  _DWORD *v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r12
  __int64 v18; // r15
  unsigned __int64 v19; // rax
  unsigned __int8 v20; // r14
  unsigned int v21; // esi
  __int64 v22; // r9
  unsigned int v23; // ecx
  _QWORD *v24; // rdi
  __int64 v25; // rdx
  int v26; // r11d
  _QWORD *v27; // r8
  int v28; // ecx
  int v29; // ecx
  int v30; // edi
  int v31; // edi
  __int64 v32; // r9
  unsigned int v33; // eax
  __int64 v34; // r11
  int v35; // esi
  _QWORD *v36; // rdx
  int v37; // r9d
  int v38; // r9d
  __int64 v39; // r10
  _QWORD *v40; // rdi
  unsigned int v41; // eax
  int v42; // edx
  __int64 v43; // rsi
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rax
  _DWORD *v46; // rax
  __int64 v47; // rdx
  _DWORD *i; // rdx
  __int64 v49; // [rsp+8h] [rbp-48h]
  unsigned __int8 v50; // [rsp+17h] [rbp-39h]
  int v51; // [rsp+18h] [rbp-38h]
  unsigned int v52; // [rsp+18h] [rbp-38h]

  *(_QWORD *)(a1 + 200) = *(_QWORD *)(a2 + 32);
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  *(_QWORD *)(a1 + 208) = v4;
  v49 = a1 + 216;
  sub_2E476F0(a1 + 216);
  sub_2E476F0(a1 + 248);
  v5 = *(_DWORD *)(a1 + 296);
  ++*(_QWORD *)(a1 + 280);
  if ( !v5 )
  {
    if ( !*(_DWORD *)(a1 + 300) )
      goto LABEL_9;
    v6 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)v6 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 288), 4 * v6, 4);
      *(_QWORD *)(a1 + 288) = 0;
      *(_QWORD *)(a1 + 296) = 0;
      *(_DWORD *)(a1 + 304) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v11 = 4 * v5;
  v6 = *(unsigned int *)(a1 + 304);
  if ( (unsigned int)(4 * v5) < 0x40 )
    v11 = 64;
  if ( v11 >= (unsigned int)v6 )
  {
LABEL_6:
    v7 = *(void **)(a1 + 288);
    if ( 4LL * (unsigned int)v6 )
      memset(v7, 255, 4LL * (unsigned int)v6);
    *(_QWORD *)(a1 + 296) = 0;
    goto LABEL_9;
  }
  v12 = v5 - 1;
  if ( v12 )
  {
    _BitScanReverse(&v12, v12);
    v13 = *(_DWORD **)(a1 + 288);
    v14 = 1 << (33 - (v12 ^ 0x1F));
    if ( v14 < 64 )
      v14 = 64;
    if ( (_DWORD)v6 == v14 )
    {
      *(_QWORD *)(a1 + 296) = 0;
      v15 = &v13[v6];
      do
      {
        if ( v13 )
          *v13 = -1;
        ++v13;
      }
      while ( v15 != v13 );
      goto LABEL_9;
    }
  }
  else
  {
    v13 = *(_DWORD **)(a1 + 288);
    v14 = 64;
  }
  sub_C7D6A0((__int64)v13, 4 * v6, 4);
  v44 = ((((((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
       | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
       | (4 * v14 / 3u + 1)
       | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 16;
  v45 = (v44
       | (((((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
           | (4 * v14 / 3u + 1)
           | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
         | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
         | (4 * v14 / 3u + 1)
         | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4)
       | (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
       | (4 * v14 / 3u + 1)
       | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 304) = v45;
  v46 = (_DWORD *)sub_C7D670(4 * v45, 4);
  v47 = *(unsigned int *)(a1 + 304);
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 288) = v46;
  for ( i = &v46[v47]; i != v46; ++v46 )
  {
    if ( v46 )
      *v46 = -1;
  }
LABEL_9:
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 184LL))(a1, a2);
  v50 = 0;
  if ( *(_DWORD *)(a1 + 264) )
  {
    v16 = *(__int64 **)(a1 + 256);
    v17 = &v16[*(unsigned int *)(a1 + 272)];
    if ( v16 != v17 )
    {
      while ( *v16 == -8192 || *v16 == -4096 )
      {
        if ( v17 == ++v16 )
        {
          v50 = 0;
          goto LABEL_10;
        }
      }
      v50 = 0;
      if ( v17 == v16 )
        goto LABEL_10;
      while ( 1 )
      {
        v18 = *v16;
        v51 = *(_DWORD *)(*(_QWORD *)(*v16 + 32) + 48LL);
        v19 = sub_2EBEE10(*(_QWORD *)(a1 + 200), v51);
        v20 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 192LL))(a1, v19);
        if ( !v20 )
          goto LABEL_41;
        sub_2EBED50(*(_QWORD *)(a1 + 200), *(_DWORD *)(*(_QWORD *)(v18 + 32) + 8LL), v51);
        v21 = *(_DWORD *)(a1 + 240);
        if ( !v21 )
          break;
        v22 = *(_QWORD *)(a1 + 224);
        v23 = (v21 - 1) & (((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9));
        v24 = (_QWORD *)(v22 + 8LL * v23);
        v25 = *v24;
        if ( v18 != *v24 )
        {
          v26 = 1;
          v27 = 0;
          while ( v25 != -4096 )
          {
            if ( v25 != -8192 || v27 )
              v24 = v27;
            v23 = (v21 - 1) & (v26 + v23);
            v25 = *(_QWORD *)(v22 + 8LL * v23);
            if ( v18 == v25 )
              goto LABEL_50;
            ++v26;
            v27 = v24;
            v24 = (_QWORD *)(v22 + 8LL * v23);
          }
          v28 = *(_DWORD *)(a1 + 232);
          if ( !v27 )
            v27 = v24;
          ++*(_QWORD *)(a1 + 216);
          v29 = v28 + 1;
          if ( 4 * v29 < 3 * v21 )
          {
            if ( v21 - *(_DWORD *)(a1 + 236) - v29 <= v21 >> 3 )
            {
              v52 = ((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9);
              sub_2E36C70(v49, v21);
              v37 = *(_DWORD *)(a1 + 240);
              if ( !v37 )
              {
LABEL_96:
                ++*(_DWORD *)(a1 + 232);
                BUG();
              }
              v38 = v37 - 1;
              v39 = *(_QWORD *)(a1 + 224);
              v40 = 0;
              v41 = v38 & v52;
              v29 = *(_DWORD *)(a1 + 232) + 1;
              v27 = (_QWORD *)(v39 + 8LL * (v38 & v52));
              v42 = 1;
              v43 = *v27;
              if ( v18 != *v27 )
              {
                while ( v43 != -4096 )
                {
                  if ( !v40 && v43 == -8192 )
                    v40 = v27;
                  v41 = v38 & (v41 + v42);
                  v27 = (_QWORD *)(v39 + 8LL * v41);
                  v43 = *v27;
                  if ( v18 == *v27 )
                    goto LABEL_58;
                  ++v42;
                }
                if ( v40 )
                  v27 = v40;
              }
            }
            goto LABEL_58;
          }
LABEL_62:
          sub_2E36C70(v49, 2 * v21);
          v30 = *(_DWORD *)(a1 + 240);
          if ( !v30 )
            goto LABEL_96;
          v31 = v30 - 1;
          v32 = *(_QWORD *)(a1 + 224);
          v33 = v31 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v29 = *(_DWORD *)(a1 + 232) + 1;
          v27 = (_QWORD *)(v32 + 8LL * v33);
          v34 = *v27;
          if ( v18 != *v27 )
          {
            v35 = 1;
            v36 = 0;
            while ( v34 != -4096 )
            {
              if ( !v36 && v34 == -8192 )
                v36 = v27;
              v33 = v31 & (v33 + v35);
              v27 = (_QWORD *)(v32 + 8LL * v33);
              v34 = *v27;
              if ( v18 == *v27 )
                goto LABEL_58;
              ++v35;
            }
            if ( v36 )
              v27 = v36;
          }
LABEL_58:
          *(_DWORD *)(a1 + 232) = v29;
          if ( *v27 != -4096 )
            --*(_DWORD *)(a1 + 236);
          *v27 = v18;
        }
LABEL_50:
        v50 = v20;
LABEL_41:
        if ( ++v16 != v17 )
        {
          while ( *v16 == -4096 || *v16 == -8192 )
          {
            if ( v17 == ++v16 )
              goto LABEL_10;
          }
          if ( v17 != v16 )
            continue;
        }
        goto LABEL_10;
      }
      ++*(_QWORD *)(a1 + 216);
      goto LABEL_62;
    }
  }
LABEL_10:
  v8 = *(__int64 **)(a1 + 224);
  v9 = &v8[*(unsigned int *)(a1 + 240)];
  if ( *(_DWORD *)(a1 + 232) && v9 != v8 )
  {
    while ( *v8 == -4096 || *v8 == -8192 )
    {
      if ( ++v8 == v9 )
        return v50;
    }
LABEL_29:
    if ( v8 != v9 )
    {
      sub_2E88E20(*v8);
      while ( ++v8 != v9 )
      {
        if ( *v8 != -8192 && *v8 != -4096 )
          goto LABEL_29;
      }
    }
  }
  return v50;
}
