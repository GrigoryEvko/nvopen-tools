// Function: sub_37F67F0
// Address: 0x37f67f0
//
__int64 __fastcall sub_37F67F0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 *v8; // rcx
  __int64 *v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 v14; // r10
  __int64 result; // rax
  __int64 *v16; // r8
  __int64 v17; // rsi
  _DWORD *v18; // rdi
  unsigned int v19; // r15d
  __int64 v20; // r14
  __int64 v21; // rcx
  __int64 *v22; // r9
  unsigned __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // r8
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // rax
  unsigned __int64 *v30; // rax
  unsigned __int64 v31; // r13
  __int64 v32; // r9
  unsigned int *v33; // r10
  unsigned int *i; // r15
  int v35; // edx
  __int16 *v36; // r13
  unsigned int v37; // r14d
  __int64 v38; // rdx
  __int64 v39; // rcx
  _DWORD *v40; // rdx
  __int64 *v41; // r11
  __int64 v42; // rax
  __int64 *v43; // r11
  unsigned __int64 v44; // r8
  unsigned __int64 v45; // rdi
  __int64 v46; // rax
  unsigned __int64 v47; // r8
  __int64 *v48; // [rsp+8h] [rbp-48h]
  unsigned __int64 v49; // [rsp+8h] [rbp-48h]
  __int64 *v50; // [rsp+10h] [rbp-40h]
  __int64 *v51; // [rsp+10h] [rbp-40h]
  __int64 *v52; // [rsp+10h] [rbp-40h]
  unsigned int *v53; // [rsp+10h] [rbp-40h]
  __int64 *v54; // [rsp+18h] [rbp-38h]
  __int64 *v55; // [rsp+18h] [rbp-38h]
  __int64 *v56; // [rsp+18h] [rbp-38h]
  unsigned int *v57; // [rsp+18h] [rbp-38h]
  unsigned int *v58; // [rsp+18h] [rbp-38h]
  unsigned __int64 v59; // [rsp+18h] [rbp-38h]

  v4 = *(unsigned int *)(a2 + 24);
  v5 = *(unsigned int *)(a1 + 304);
  v6 = 24 * v4;
  v7 = 24 * v4 + *(_QWORD *)(a1 + 496);
  v8 = *(__int64 **)(v7 + 8);
  v9 = *(__int64 **)v7;
  v10 = ((__int64)v8 - *(_QWORD *)v7) >> 3;
  if ( v5 > v10 )
  {
    sub_37F6380((__int64 **)v7, v5 - v10);
  }
  else if ( v5 < v10 )
  {
    v48 = &v9[v5];
    if ( v8 != v48 )
    {
      v28 = &v9[v5];
      do
      {
        v29 = *v28;
        if ( *v28 )
        {
          if ( (v29 & 1) != 0 )
          {
            v30 = (unsigned __int64 *)(v29 & 0xFFFFFFFFFFFFFFFELL);
            v31 = (unsigned __int64)v30;
            if ( v30 )
            {
              if ( (unsigned __int64 *)*v30 != v30 + 2 )
              {
                v50 = v28;
                v55 = v8;
                _libc_free(*v30);
                v28 = v50;
                v8 = v55;
              }
              v51 = v28;
              v56 = v8;
              j_j___libc_free_0(v31);
              v28 = v51;
              v8 = v56;
            }
          }
        }
        ++v28;
      }
      while ( v8 != v28 );
      *(_QWORD *)(v7 + 8) = v48;
    }
  }
  *(_DWORD *)(a1 + 456) = 0;
  if ( *(_QWORD *)(a1 + 320) == *(_QWORD *)(a1 + 328) )
    sub_37F6680(a1 + 320, *(unsigned int *)(a1 + 304), (int *)(a1 + 640));
  v11 = *(unsigned int *)(a2 + 72);
  if ( (_DWORD)v11 )
  {
    v12 = *(_QWORD *)(a2 + 64);
    v13 = *(_DWORD *)(a1 + 304);
    v14 = v12 + 8 * v11;
    do
    {
      result = *(_QWORD *)(a1 + 344);
      v16 = (__int64 *)(result + 24LL * *(int *)(*(_QWORD *)v12 + 24LL));
      v17 = *v16;
      if ( v16[1] != *v16 && v13 )
      {
        v13 = 0;
        while ( 1 )
        {
          v18 = (_DWORD *)(*(_QWORD *)(a1 + 320) + 4LL * v13);
          result = (unsigned int)*v18;
          if ( *(_DWORD *)(v17 + 4LL * v13) >= (int)result )
            result = *(unsigned int *)(v17 + 4LL * v13);
          ++v13;
          *v18 = result;
          if ( *(_DWORD *)(a1 + 304) == v13 )
            break;
          v17 = *v16;
        }
      }
      v12 += 8;
    }
    while ( v14 != v12 );
    v19 = 0;
    if ( !v13 )
      return result;
    while ( 1 )
    {
      while ( 1 )
      {
        result = *(int *)(*(_QWORD *)(a1 + 320) + 4LL * v19);
        if ( (_DWORD)result != *(_DWORD *)(a1 + 640) )
          break;
LABEL_21:
        if ( *(_DWORD *)(a1 + 304) == ++v19 )
          return result;
      }
      v20 = 4 * result + 2;
      v21 = *(_QWORD *)(*(_QWORD *)(a1 + 496) + v6);
      v22 = (__int64 *)(v21 + 8LL * v19);
      result = *v22;
      v23 = *v22 & 0xFFFFFFFFFFFFFFFELL;
      if ( v23 )
      {
        if ( (result & 1) == 0 )
        {
          v54 = (__int64 *)(v21 + 8LL * v19);
          v24 = sub_22077B0(0x30u);
          v22 = v54;
          if ( v24 )
          {
            *(_QWORD *)v24 = v24 + 16;
            *(_QWORD *)(v24 + 8) = 0x400000000LL;
          }
          v26 = v24 & 0xFFFFFFFFFFFFFFFELL;
          *v54 = v24 | 1;
          v27 = *(unsigned int *)((v24 & 0xFFFFFFFFFFFFFFFELL) + 8);
          if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v26 + 12) )
          {
            sub_C8D5F0(v26, (const void *)(v26 + 16), v27 + 1, 8u, v25, (__int64)v54);
            v22 = v54;
            v27 = *(unsigned int *)(v26 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v26 + 8 * v27) = v23;
          ++*(_DWORD *)(v26 + 8);
          v16 = (__int64 *)(*v22 & 0xFFFFFFFFFFFFFFFELL);
          v23 = (unsigned __int64)v16;
        }
        result = *(unsigned int *)(v23 + 8);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(v23 + 12) )
        {
          sub_C8D5F0(v23, (const void *)(v23 + 16), result + 1, 8u, (__int64)v16, (__int64)v22);
          result = *(unsigned int *)(v23 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v23 + 8 * result) = v20;
        ++*(_DWORD *)(v23 + 8);
        goto LABEL_21;
      }
      *v22 = v20;
      if ( *(_DWORD *)(a1 + 304) == ++v19 )
        return result;
    }
  }
  v57 = *(unsigned int **)(a2 + 192);
  result = sub_2E33140(a2);
  v33 = v57;
  for ( i = (unsigned int *)result; v33 != i; i += 6 )
  {
    v38 = *(_QWORD *)(a1 + 208);
    v39 = *(_QWORD *)(v38 + 8);
    result = *(_DWORD *)(v39 + 24LL * *i + 16) >> 12;
    v37 = *(_DWORD *)(v39 + 24LL * *i + 16) & 0xFFF;
    v36 = (__int16 *)(*(_QWORD *)(v38 + 56) + 2 * result);
    do
    {
      if ( !v36 )
        break;
      result = v37;
      v40 = (_DWORD *)(*(_QWORD *)(a1 + 320) + 4LL * v37);
      if ( *v40 != -1 )
      {
        *v40 = -1;
        v41 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 496) + v6) + 8LL * v37);
        result = *v41;
        v47 = *v41 & 0xFFFFFFFFFFFFFFFELL;
        if ( v47 )
        {
          if ( (result & 1) == 0 )
          {
            v49 = *v41 & 0xFFFFFFFFFFFFFFFELL;
            v52 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 496) + v6) + 8LL * v37);
            v58 = v33;
            v42 = sub_22077B0(0x30u);
            v33 = v58;
            v43 = v52;
            v44 = v49;
            if ( v42 )
            {
              *(_QWORD *)v42 = v42 + 16;
              *(_QWORD *)(v42 + 8) = 0x400000000LL;
            }
            v45 = v42 & 0xFFFFFFFFFFFFFFFELL;
            *v52 = v42 | 1;
            v46 = *(unsigned int *)((v42 & 0xFFFFFFFFFFFFFFFELL) + 8);
            if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v45 + 12) )
            {
              sub_C8D5F0(v45, (const void *)(v45 + 16), v46 + 1, 8u, v49, v32);
              v44 = v49;
              v43 = v52;
              v33 = v58;
              v46 = *(unsigned int *)(v45 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v45 + 8 * v46) = v44;
            ++*(_DWORD *)(v45 + 8);
            v47 = *v43 & 0xFFFFFFFFFFFFFFFELL;
          }
          result = *(unsigned int *)(v47 + 8);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(v47 + 12) )
          {
            v53 = v33;
            v59 = v47;
            sub_C8D5F0(v47, (const void *)(v47 + 16), result + 1, 8u, v47, v32);
            v47 = v59;
            v33 = v53;
            result = *(unsigned int *)(v59 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v47 + 8 * result) = -2;
          ++*(_DWORD *)(v47 + 8);
        }
        else
        {
          *v41 = -2;
        }
      }
      v35 = *v36++;
      v37 += v35;
    }
    while ( (_WORD)v35 );
  }
  return result;
}
