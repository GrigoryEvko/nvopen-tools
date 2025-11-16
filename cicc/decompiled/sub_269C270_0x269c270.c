// Function: sub_269C270
// Address: 0x269c270
//
__int64 __fastcall sub_269C270(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // rdi
  __int64 *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 result; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r12
  _QWORD *v15; // r12
  _QWORD *v16; // r13
  __int64 v17; // r8
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  __int64 v20; // rcx
  unsigned int v21; // esi
  int v22; // eax
  int v23; // ecx
  __int64 v24; // r8
  unsigned int v25; // eax
  _QWORD *v26; // r10
  __int64 v27; // rdi
  int v28; // edx
  int v29; // r11d
  int v30; // eax
  int v31; // eax
  int v32; // ecx
  __int64 v33; // r8
  _QWORD *v34; // r9
  int v35; // r11d
  unsigned int v36; // eax
  __int64 v37; // rdi
  int v38; // r11d
  unsigned __int8 v39; // [rsp+Fh] [rbp-51h]
  _BYTE v40[32]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 v41; // [rsp+30h] [rbp-30h]

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v4 = *(_QWORD **)(a1 + 32);
    v6 = &v4[*(unsigned int *)(a1 + 40)];
    if ( v6 != sub_266E4D0(v4, (__int64)v6, a2) )
      return 0;
    v10 = *a2;
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v7 + 1, 8u, v7, v8);
      v6 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
    }
    *v6 = v10;
    v11 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
    *(_DWORD *)(a1 + 40) = v11;
    if ( (unsigned int)v11 <= 0x10 )
      return 1;
    v15 = *(_QWORD **)(a1 + 32);
    v16 = &v15[v11];
    while ( 1 )
    {
      v21 = *(_DWORD *)(a1 + 24);
      if ( !v21 )
        break;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = (v21 - 1) & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
      v19 = (_QWORD *)(v17 + 8LL * v18);
      v20 = *v19;
      if ( *v15 != *v19 )
      {
        v29 = 1;
        v26 = 0;
        while ( v20 != -4096 )
        {
          if ( v26 || v20 != -8192 )
            v19 = v26;
          v18 = (v21 - 1) & (v29 + v18);
          v20 = *(_QWORD *)(v17 + 8LL * v18);
          if ( *v15 == v20 )
            goto LABEL_14;
          ++v29;
          v26 = v19;
          v19 = (_QWORD *)(v17 + 8LL * v18);
        }
        v30 = *(_DWORD *)(a1 + 16);
        if ( !v26 )
          v26 = v19;
        ++*(_QWORD *)a1;
        v28 = v30 + 1;
        if ( 4 * (v30 + 1) < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(a1 + 20) - v28 <= v21 >> 3 )
          {
            sub_24FB720(a1, v21);
            v31 = *(_DWORD *)(a1 + 24);
            if ( !v31 )
            {
LABEL_51:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v32 = v31 - 1;
            v33 = *(_QWORD *)(a1 + 8);
            v34 = 0;
            v35 = 1;
            v36 = (v31 - 1) & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
            v26 = (_QWORD *)(v33 + 8LL * v36);
            v37 = *v26;
            v28 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v26 != *v15 )
            {
              while ( v37 != -4096 )
              {
                if ( v37 == -8192 && !v34 )
                  v34 = v26;
                v36 = v32 & (v35 + v36);
                v26 = (_QWORD *)(v33 + 8LL * v36);
                v37 = *v26;
                if ( *v15 == *v26 )
                  goto LABEL_19;
                ++v35;
              }
LABEL_31:
              if ( v34 )
                v26 = v34;
            }
          }
LABEL_19:
          *(_DWORD *)(a1 + 16) = v28;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v26 = *v15;
          goto LABEL_14;
        }
LABEL_17:
        sub_24FB720(a1, 2 * v21);
        v22 = *(_DWORD *)(a1 + 24);
        if ( !v22 )
          goto LABEL_51;
        v23 = v22 - 1;
        v24 = *(_QWORD *)(a1 + 8);
        v25 = (v22 - 1) & (((unsigned int)*v15 >> 9) ^ ((unsigned int)*v15 >> 4));
        v26 = (_QWORD *)(v24 + 8LL * v25);
        v27 = *v26;
        v28 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v26 != *v15 )
        {
          v38 = 1;
          v34 = 0;
          while ( v27 != -4096 )
          {
            if ( v27 == -8192 && !v34 )
              v34 = v26;
            v25 = v23 & (v38 + v25);
            v26 = (_QWORD *)(v24 + 8LL * v25);
            v27 = *v26;
            if ( *v15 == *v26 )
              goto LABEL_19;
            ++v38;
          }
          goto LABEL_31;
        }
        goto LABEL_19;
      }
LABEL_14:
      if ( v16 == ++v15 )
        return 1;
    }
    ++*(_QWORD *)a1;
    goto LABEL_17;
  }
  sub_269BA20((__int64)v40, a1, a2);
  result = v41;
  if ( !v41 )
    return 0;
  v13 = *(unsigned int *)(a1 + 40);
  v14 = *a2;
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v39 = v41;
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v13 + 1, 8u, v13 + 1, v12);
    v13 = *(unsigned int *)(a1 + 40);
    result = v39;
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v13) = v14;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
