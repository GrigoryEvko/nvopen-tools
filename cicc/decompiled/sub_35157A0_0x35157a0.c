// Function: sub_35157A0
// Address: 0x35157a0
//
unsigned __int64 __fastcall sub_35157A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *i; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r11
  unsigned __int64 result; // rax
  char v13; // dl
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // r15
  _QWORD *v17; // rdi
  _QWORD *v18; // rsi
  unsigned int v19; // esi
  int v20; // r11d
  _QWORD *v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rbx
  int v25; // eax
  __int64 v26; // rcx
  int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // edi
  int v31; // eax
  int v32; // eax
  __int64 v33; // rbx
  int v34; // edx
  int v35; // edx
  unsigned int v36; // esi
  __int64 v37; // rdi
  _QWORD *v38; // r11
  int v39; // esi
  int v40; // esi
  int v41; // r11d
  unsigned int v42; // edx
  __int64 v43; // rdi
  unsigned int v44; // [rsp+4h] [rbp-6Ch]
  __int64 v45; // [rsp+8h] [rbp-68h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  _QWORD *v47; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 *v49; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+28h] [rbp-48h] BYREF
  __int64 v51[7]; // [rsp+38h] [rbp-38h] BYREF

  v50 = a2;
  v45 = a1 + 888;
  v11 = *sub_3515040(a1 + 888, &v50);
  if ( !*(_BYTE *)(a3 + 28) )
    goto LABEL_8;
  result = *(_QWORD *)(a3 + 8);
  v8 = *(unsigned int *)(a3 + 20);
  for ( i = (__int64 *)(result + 8 * v8); i != (__int64 *)result; result += 8LL )
  {
    if ( v11 == *(_QWORD *)result )
      return result;
  }
  if ( (unsigned int)v8 < *(_DWORD *)(a3 + 16) )
  {
    *(_DWORD *)(a3 + 20) = v8 + 1;
    *i = v11;
    ++*(_QWORD *)a3;
  }
  else
  {
LABEL_8:
    v48 = v11;
    result = (unsigned __int64)sub_C8CC70(a3, v11, (__int64)i, v8, v9, v10);
    v11 = v48;
    if ( !v13 )
      return result;
  }
  result = *(_QWORD *)v11;
  v47 = *(_QWORD **)v11;
  v46 = *(_QWORD *)v11 + 8LL * *(unsigned int *)(v11 + 8);
  if ( *(_QWORD *)v11 == v46 )
    goto LABEL_41;
  do
  {
    v14 = *(__int64 **)(*v47 + 64LL);
    v15 = *(unsigned int *)(*v47 + 72LL);
    if ( v14 == &v14[v15] )
      goto LABEL_40;
    v49 = &v14[v15];
    v16 = v11;
    do
    {
      v24 = *v14;
      v51[0] = *v14;
      if ( !a4 )
        goto LABEL_13;
      if ( !*(_DWORD *)(a4 + 16) )
      {
        v17 = *(_QWORD **)(a4 + 32);
        v18 = &v17[*(unsigned int *)(a4 + 40)];
        if ( v18 == sub_3510810(v17, (__int64)v18, v51) )
          goto LABEL_17;
LABEL_13:
        v19 = *(_DWORD *)(a1 + 912);
        if ( v19 )
        {
          v10 = *(_QWORD *)(a1 + 896);
          v20 = 1;
          v21 = 0;
          v9 = (v19 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v22 = (_QWORD *)(v10 + 16 * v9);
          v23 = *v22;
          if ( v24 == *v22 )
          {
LABEL_15:
            if ( v16 == v22[1] )
              goto LABEL_17;
            goto LABEL_16;
          }
          while ( v23 != -4096 )
          {
            if ( !v21 && v23 == -8192 )
              v21 = v22;
            v9 = (v19 - 1) & (v20 + (_DWORD)v9);
            v22 = (_QWORD *)(v10 + 16LL * (unsigned int)v9);
            v23 = *v22;
            if ( v24 == *v22 )
              goto LABEL_15;
            ++v20;
          }
          if ( !v21 )
            v21 = v22;
          v31 = *(_DWORD *)(a1 + 904);
          ++*(_QWORD *)(a1 + 888);
          v32 = v31 + 1;
          if ( 4 * v32 < 3 * v19 )
          {
            v9 = v19 >> 3;
            if ( v19 - *(_DWORD *)(a1 + 908) - v32 <= (unsigned int)v9 )
            {
              v44 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
              sub_3512300(v45, v19);
              v39 = *(_DWORD *)(a1 + 912);
              if ( !v39 )
              {
LABEL_74:
                ++*(_DWORD *)(a1 + 904);
                BUG();
              }
              v40 = v39 - 1;
              v9 = *(_QWORD *)(a1 + 896);
              v10 = 0;
              v41 = 1;
              v42 = v40 & v44;
              v32 = *(_DWORD *)(a1 + 904) + 1;
              v21 = (_QWORD *)(v9 + 16LL * (v40 & v44));
              v43 = *v21;
              if ( v24 != *v21 )
              {
                while ( v43 != -4096 )
                {
                  if ( v43 == -8192 && !v10 )
                    v10 = (__int64)v21;
                  v42 = v40 & (v41 + v42);
                  v21 = (_QWORD *)(v9 + 16LL * v42);
                  v43 = *v21;
                  if ( v24 == *v21 )
                    goto LABEL_36;
                  ++v41;
                }
                if ( v10 )
                  v21 = (_QWORD *)v10;
              }
            }
            goto LABEL_36;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 888);
        }
        sub_3512300(v45, 2 * v19);
        v34 = *(_DWORD *)(a1 + 912);
        if ( !v34 )
          goto LABEL_74;
        v35 = v34 - 1;
        v9 = *(_QWORD *)(a1 + 896);
        v36 = v35 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v32 = *(_DWORD *)(a1 + 904) + 1;
        v21 = (_QWORD *)(v9 + 16LL * v36);
        v37 = *v21;
        if ( v24 != *v21 )
        {
          v10 = 1;
          v38 = 0;
          while ( v37 != -4096 )
          {
            if ( v37 == -8192 && !v38 )
              v38 = v21;
            v36 = v35 & (v10 + v36);
            v21 = (_QWORD *)(v9 + 16LL * v36);
            v37 = *v21;
            if ( v24 == *v21 )
              goto LABEL_36;
            v10 = (unsigned int)(v10 + 1);
          }
          if ( v38 )
            v21 = v38;
        }
LABEL_36:
        *(_DWORD *)(a1 + 904) = v32;
        if ( *v21 != -4096 )
          --*(_DWORD *)(a1 + 908);
        *v21 = v24;
        v21[1] = 0;
LABEL_16:
        ++*(_DWORD *)(v16 + 56);
        goto LABEL_17;
      }
      v25 = *(_DWORD *)(a4 + 24);
      v26 = *(_QWORD *)(a4 + 8);
      if ( v25 )
      {
        v27 = v25 - 1;
        v28 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v29 = *(_QWORD *)(v26 + 8LL * v28);
        if ( v24 == v29 )
          goto LABEL_13;
        v30 = 1;
        while ( v29 != -4096 )
        {
          v9 = (unsigned int)(v30 + 1);
          v28 = v27 & (v30 + v28);
          v29 = *(_QWORD *)(v26 + 8LL * v28);
          if ( v24 == v29 )
            goto LABEL_13;
          ++v30;
        }
      }
LABEL_17:
      ++v14;
    }
    while ( v49 != v14 );
    v11 = v16;
LABEL_40:
    result = (unsigned __int64)++v47;
  }
  while ( (_QWORD *)v46 != v47 );
LABEL_41:
  if ( !*(_DWORD *)(v11 + 56) )
  {
    v33 = **(_QWORD **)v11;
    if ( *(_BYTE *)(v33 + 216) )
    {
      result = *(unsigned int *)(a1 + 352);
      if ( result + 1 > *(unsigned int *)(a1 + 356) )
      {
        sub_C8D5F0(a1 + 344, (const void *)(a1 + 360), result + 1, 8u, v9, v10);
        result = *(unsigned int *)(a1 + 352);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 344) + 8 * result) = v33;
      ++*(_DWORD *)(a1 + 352);
    }
    else
    {
      result = *(unsigned int *)(a1 + 208);
      if ( result + 1 > *(unsigned int *)(a1 + 212) )
      {
        sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), result + 1, 8u, v9, v10);
        result = *(unsigned int *)(a1 + 208);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * result) = v33;
      ++*(_DWORD *)(a1 + 208);
    }
  }
  return result;
}
