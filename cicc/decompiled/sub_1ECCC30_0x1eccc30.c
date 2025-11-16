// Function: sub_1ECCC30
// Address: 0x1eccc30
//
unsigned __int64 __fastcall sub_1ECCC30(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned int v5; // r12d
  int *v6; // r8
  unsigned int v7; // ebx
  _DWORD *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // rdx
  void *v11; // rax
  __int64 v12; // r8
  void *v13; // rdi
  void *v14; // rax
  __int64 v15; // r9
  _DWORD *v16; // rsi
  __int64 v17; // rax
  unsigned __int64 result; // rax
  _DWORD *v19; // rsi
  unsigned int v20; // r8d
  int v21; // r9d
  __int64 v22; // r14
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // r10
  __int64 v26; // r11
  int v27; // esi
  __int64 v28; // rbx
  unsigned int i; // edx
  __int64 v30; // rsi
  unsigned int *v31; // r12
  __int64 v32; // r10
  __int64 v33; // rdx
  __int64 v34; // rbx
  unsigned int v35; // edx
  __int64 v36; // rsi
  _DWORD *v37; // rsi
  bool v38; // zf
  int v39; // [rsp+4h] [rbp-5Ch]
  __int64 v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+10h] [rbp-50h]
  __int64 n; // [rsp+18h] [rbp-48h]
  unsigned int v43; // [rsp+24h] [rbp-3Ch] BYREF
  unsigned int v44[14]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 168);
  *(_QWORD *)v44 = a1;
  v4 = v3 - *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 152) = a2;
  v43 = 0;
  v5 = -1171354717 * (v4 >> 3);
  if ( v5 )
  {
    v6 = (int *)&v43;
    while ( 1 )
    {
      v8 = *(_DWORD **)(a1 + 192);
      if ( v8 == sub_1ECAFE0(*(_DWORD **)(a1 + 184), (__int64)v8, v6) )
        break;
      v7 = v43 + 1;
      v43 = v7;
      if ( v5 <= v7 )
        goto LABEL_6;
    }
    v7 = v43;
  }
  else
  {
    v7 = 0;
  }
LABEL_6:
  v39 = sub_1ECCC00((__int64)v44);
  while ( v39 != v7 )
  {
    v9 = *(_QWORD *)(**(_QWORD **)(a1 + 152) + 160LL) + 88LL * v7;
    v40 = v9;
    v10 = (unsigned int)(**(_DWORD **)v9 - 1);
    *(_DWORD *)(v9 + 20) = v10;
    v41 = v10;
    n = 4 * v10;
    v11 = (void *)sub_2207820(4 * v10);
    v12 = v40;
    v13 = v11;
    if ( v11 && v41 )
    {
      v14 = memset(v11, 0, n);
      v12 = v40;
      v13 = v14;
    }
    v15 = *(_QWORD *)(v12 + 32);
    *(_QWORD *)(v12 + 32) = v13;
    if ( v15 )
      j_j___libc_free_0_0(v15);
    v43 = ++v7;
    if ( v5 > v7 )
    {
      while ( 1 )
      {
        v16 = *(_DWORD **)(a1 + 192);
        if ( v16 == sub_1ECAFE0(*(_DWORD **)(a1 + 184), (__int64)v16, (int *)&v43) )
          break;
        v7 = v43 + 1;
        v43 = v7;
        if ( v5 <= v7 )
          goto LABEL_17;
      }
      v7 = v43;
    }
LABEL_17:
    ;
  }
  v17 = *(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208);
  v44[0] = 0;
  result = 0xAAAAAAAAAAAAAAABLL * (v17 >> 4);
  if ( (_DWORD)result )
  {
    do
    {
      v19 = *(_DWORD **)(a1 + 240);
      if ( v19 == sub_1ECAFE0(*(_DWORD **)(a1 + 232), (__int64)v19, (int *)v44) )
      {
        result = v44[0];
        v21 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 4);
        goto LABEL_23;
      }
      result = v44[0] + 1;
      v44[0] = result;
    }
    while ( v20 > (unsigned int)result );
    v21 = -1431655765 * ((__int64)(*(_QWORD *)(a1 + 216) - *(_QWORD *)(a1 + 208)) >> 4);
LABEL_23:
    while ( 2 )
    {
      if ( (_DWORD)result == v21 )
        return result;
      while ( 1 )
      {
        v22 = *(_QWORD *)(a1 + 152);
        v23 = 48LL * (unsigned int)result;
        v24 = v23 + *(_QWORD *)(*(_QWORD *)v22 + 208LL);
        v25 = *(_QWORD *)(*(_QWORD *)v22 + 160LL) + 88LL * *(unsigned int *)(v24 + 20);
        v26 = *(_QWORD *)v24;
        v27 = *(_DWORD *)(v25 + 24);
        if ( *(_DWORD *)(v24 + 20) == *(_DWORD *)(v24 + 24) )
        {
          *(_DWORD *)(v25 + 24) = *(_DWORD *)(v26 + 16) + v27;
          v28 = *(_QWORD *)(v26 + 32);
        }
        else
        {
          *(_DWORD *)(v25 + 24) = *(_DWORD *)(v26 + 20) + v27;
          v28 = *(_QWORD *)(v26 + 24);
        }
        for ( i = 0;
              *(_DWORD *)(v25 + 20) > i;
              *(_DWORD *)(*(_QWORD *)(v25 + 32) + 4 * v30) += *(unsigned __int8 *)(v28 + v30) )
        {
          v30 = i++;
        }
        v31 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)v22 + 208LL) + v23);
        v32 = *(_QWORD *)(*(_QWORD *)v22 + 160LL) + 88LL * v31[6];
        v33 = *(_QWORD *)v31;
        *(_DWORD *)(v32 + 24) += *(_DWORD *)(*(_QWORD *)v31 + 16LL);
        v34 = *(_QWORD *)(v33 + 32);
        if ( *(_DWORD *)(v32 + 20) )
          break;
        while ( 1 )
        {
          result = (unsigned int)(result + 1);
          v44[0] = result;
          if ( v20 <= (unsigned int)result )
            break;
LABEL_32:
          v37 = *(_DWORD **)(a1 + 240);
          v38 = v37 == sub_1ECAFE0(*(_DWORD **)(a1 + 232), (__int64)v37, (int *)v44);
          result = v44[0];
          if ( v38 )
            goto LABEL_23;
        }
        if ( (_DWORD)result == v21 )
          return result;
      }
      v35 = 0;
      do
      {
        v36 = v35++;
        *(_DWORD *)(*(_QWORD *)(v32 + 32) + 4 * v36) += *(unsigned __int8 *)(v34 + v36);
      }
      while ( *(_DWORD *)(v32 + 20) > v35 );
      result = (unsigned int)(result + 1);
      v44[0] = result;
      if ( v20 <= (unsigned int)result )
        continue;
      goto LABEL_32;
    }
  }
  return result;
}
