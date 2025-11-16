// Function: sub_1FD47F0
// Address: 0x1fd47f0
//
__int64 __fastcall sub_1FD47F0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(); // rax
  unsigned int v4; // r11d
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r10
  int v11; // r8d
  unsigned int v12; // edx
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v15; // r12
  unsigned int v16; // esi
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // r11
  int v24; // edi
  int v25; // edx
  int v26; // ecx
  int v27; // ecx
  __int64 v28; // r8
  unsigned int v29; // esi
  __int64 v30; // rdi
  int v31; // r15d
  __int64 *v32; // r9
  int v33; // ecx
  int v34; // ecx
  __int64 *v35; // r8
  int v36; // r9d
  unsigned int v37; // r15d
  __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 *v40; // rax
  int v41; // [rsp-48h] [rbp-48h]
  __int64 v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-48h] [rbp-48h]
  unsigned __int8 v44; // [rsp-39h] [rbp-39h]
  unsigned __int8 v45; // [rsp-39h] [rbp-39h]

  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 40) + 40LL) )
    return 0;
  v3 = *(__int64 (**)())(*(_QWORD *)a1 + 32LL);
  if ( v3 == sub_1FD3490 )
    return 0;
  v4 = v3();
  if ( !(_BYTE)v4 )
    return 0;
  v6 = *(__int64 **)(a1 + 40);
  v7 = *v6;
  if ( (*(_BYTE *)(*v6 + 18) & 1) != 0 )
  {
    v45 = v4;
    sub_15E08E0(*v6, a2);
    v40 = *(__int64 **)(a1 + 40);
    v8 = *(_QWORD *)(v7 + 88);
    v4 = v45;
    v7 = *v40;
    if ( (*(_BYTE *)(*v40 + 18) & 1) != 0 )
    {
      sub_15E08E0(*v40, a2);
      v9 = *(_QWORD *)(v7 + 88);
      v4 = v45;
    }
    else
    {
      v9 = *(_QWORD *)(v7 + 88);
    }
  }
  else
  {
    v8 = *(_QWORD *)(v7 + 88);
    v9 = v8;
  }
  v10 = v9 + 40LL * *(_QWORD *)(v7 + 96);
  if ( v10 != v8 )
  {
    v44 = v4;
    while ( 1 )
    {
      v21 = *(unsigned int *)(a1 + 32);
      v22 = *(_QWORD *)(a1 + 16);
      if ( (_DWORD)v21 )
      {
        v11 = 1;
        v12 = (v21 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v13 = (__int64 *)(v22 + 16LL * v12);
        v14 = *v13;
        if ( v8 != *v13 )
        {
          while ( v14 != -8 )
          {
            v12 = (v21 - 1) & (v11 + v12);
            v13 = (__int64 *)(v22 + 16LL * v12);
            v14 = *v13;
            if ( *v13 == v8 )
              goto LABEL_12;
            ++v11;
          }
          v13 = (__int64 *)(v22 + 16 * v21);
        }
      }
      else
      {
        v13 = (__int64 *)(v22 + 16 * v21);
      }
LABEL_12:
      v15 = *(_QWORD *)(a1 + 40);
      v16 = *(_DWORD *)(v15 + 232);
      if ( !v16 )
        break;
      v17 = *(_QWORD *)(v15 + 216);
      v18 = (v16 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v19;
      if ( v8 != *v19 )
      {
        v41 = 1;
        v23 = 0;
        while ( v20 != -8 )
        {
          if ( !v23 && v20 == -16 )
            v23 = v19;
          v18 = (v16 - 1) & (v41 + v18);
          v19 = (__int64 *)(v17 + 16LL * v18);
          v20 = *v19;
          if ( *v19 == v8 )
            goto LABEL_14;
          ++v41;
        }
        v24 = *(_DWORD *)(v15 + 224);
        if ( v23 )
          v19 = v23;
        ++*(_QWORD *)(v15 + 208);
        v25 = v24 + 1;
        if ( 4 * (v24 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(v15 + 228) - v25 <= v16 >> 3 )
          {
            v43 = v10;
            sub_1542080(v15 + 208, v16);
            v33 = *(_DWORD *)(v15 + 232);
            if ( !v33 )
            {
LABEL_63:
              ++*(_DWORD *)(v15 + 224);
              BUG();
            }
            v34 = v33 - 1;
            v35 = 0;
            v10 = v43;
            v36 = 1;
            v37 = v34 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v38 = *(_QWORD *)(v15 + 216);
            v25 = *(_DWORD *)(v15 + 224) + 1;
            v19 = (__int64 *)(v38 + 16LL * v37);
            v39 = *v19;
            if ( v8 != *v19 )
            {
              while ( v39 != -8 )
              {
                if ( !v35 && v39 == -16 )
                  v35 = v19;
                v37 = v34 & (v36 + v37);
                v19 = (__int64 *)(v38 + 16LL * v37);
                v39 = *v19;
                if ( v8 == *v19 )
                  goto LABEL_28;
                ++v36;
              }
              if ( v35 )
                v19 = v35;
            }
          }
          goto LABEL_28;
        }
LABEL_32:
        v42 = v10;
        sub_1542080(v15 + 208, 2 * v16);
        v26 = *(_DWORD *)(v15 + 232);
        if ( !v26 )
          goto LABEL_63;
        v27 = v26 - 1;
        v28 = *(_QWORD *)(v15 + 216);
        v10 = v42;
        v29 = v27 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v25 = *(_DWORD *)(v15 + 224) + 1;
        v19 = (__int64 *)(v28 + 16LL * v29);
        v30 = *v19;
        if ( *v19 != v8 )
        {
          v31 = 1;
          v32 = 0;
          while ( v30 != -8 )
          {
            if ( v30 == -16 && !v32 )
              v32 = v19;
            v29 = v27 & (v31 + v29);
            v19 = (__int64 *)(v28 + 16LL * v29);
            v30 = *v19;
            if ( *v19 == v8 )
              goto LABEL_28;
            ++v31;
          }
          if ( v32 )
            v19 = v32;
        }
LABEL_28:
        *(_DWORD *)(v15 + 224) = v25;
        if ( *v19 != -8 )
          --*(_DWORD *)(v15 + 228);
        *v19 = v8;
        *((_DWORD *)v19 + 2) = 0;
      }
LABEL_14:
      v8 += 40;
      *((_DWORD *)v19 + 2) = *((_DWORD *)v13 + 2);
      if ( v8 == v10 )
        return v44;
    }
    ++*(_QWORD *)(v15 + 208);
    goto LABEL_32;
  }
  return v4;
}
