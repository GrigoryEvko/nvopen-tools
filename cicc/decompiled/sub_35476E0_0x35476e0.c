// Function: sub_35476E0
// Address: 0x35476e0
//
__int64 __fastcall sub_35476E0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  int v12; // ebx
  int v13; // r14d
  __int64 v14; // rax
  __int64 v15; // r13
  unsigned __int64 v16; // r12
  int v17; // eax
  __int64 v18; // r13
  __int64 v19; // rax
  int v20; // r14d
  _DWORD *v21; // r13
  int v22; // r12d
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // eax
  __int64 v26; // r13
  __int64 v27; // r13
  __int64 result; // rax
  __int64 *v29; // r14
  __int64 *j; // rbx
  __int64 v31; // r12
  _DWORD *v32; // rdx
  int v33; // eax
  __int64 v34; // rdx
  int *v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+8h] [rbp-58h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+10h] [rbp-50h]
  int v40; // [rsp+1Ch] [rbp-44h]
  int *v41; // [rsp+20h] [rbp-40h]
  __int64 i; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+28h] [rbp-38h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  __int64 v45; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 3952);
  v9 = *(_QWORD *)(a1 + 3944);
  v10 = v7 >> 8;
  v11 = (v8 - v9) >> 4;
  if ( v10 > v11 )
  {
    v10 -= v11;
    sub_3547520((const __m128i **)(a1 + 3944), v10);
  }
  else if ( v10 < v11 )
  {
    v10 = v9 + 16 * v10;
    if ( v8 != v10 )
      *(_QWORD *)(a1 + 3952) = v10;
  }
  v36 = *(int **)(a1 + 3832);
  if ( *(int **)(a1 + 3824) != v36 )
  {
    v41 = *(int **)(a1 + 3824);
    v40 = 0;
    do
    {
      v12 = 0;
      v13 = 0;
      v38 = *v41;
      v10 = *(_QWORD *)(a1 + 48) + (v38 << 8);
      v14 = sub_35459D0(*(_QWORD **)(a1 + 3464), v10);
      v15 = *(_QWORD *)v14;
      v43 = *(_QWORD *)v14 + 32LL * *(unsigned int *)(v14 + 8);
      if ( *(_QWORD *)v14 != v43 )
      {
        do
        {
          v16 = *(_QWORD *)(v15 + 8) & 0xFFFFFFFFFFFFFFF8LL;
          if ( !*(_DWORD *)(v15 + 20)
            && v12 < *(_DWORD *)(*(_QWORD *)(a1 + 3944) + 16LL * *(unsigned int *)(v16 + 200) + 8) + 1 )
          {
            v12 = *(_DWORD *)(*(_QWORD *)(a1 + 3944) + 16LL * *(unsigned int *)(v16 + 200) + 8) + 1;
          }
          v10 = 1;
          if ( !(unsigned __int8)sub_3545640(v15, 1u) )
          {
            v10 = *(_QWORD *)(a1 + 3944) + 16LL * *(unsigned int *)(v16 + 200);
            if ( v13 < *(_DWORD *)v10 + *(_DWORD *)(v15 + 20) - *(_DWORD *)(v15 + 24) * *(_DWORD *)(a1 + 3472) )
              v13 = *(_DWORD *)v10 + *(_DWORD *)(v15 + 20) - *(_DWORD *)(v15 + 24) * *(_DWORD *)(a1 + 3472);
          }
          v15 += 32;
        }
        while ( v43 != v15 );
      }
      v17 = v40;
      if ( v40 < v13 )
        v17 = v13;
      v18 = 16 * v38;
      ++v41;
      v40 = v17;
      *(_DWORD *)(*(_QWORD *)(a1 + 3944) + v18) = v13;
      *(_DWORD *)(*(_QWORD *)(a1 + 3944) + v18 + 8) = v12;
    }
    while ( v36 != v41 );
    a4 = *(_QWORD *)(a1 + 3832);
    v37 = *(_QWORD *)(a1 + 3824);
    for ( i = a4; v37 != i; *(_DWORD *)(*(_QWORD *)(a1 + 3944) + v26 + 12) = v22 )
    {
      v39 = *(int *)(i - 4);
      v10 = *(_QWORD *)(a1 + 48) + (v39 << 8);
      v19 = sub_3545E90(*(_QWORD **)(a1 + 3464), v10);
      v44 = *(_QWORD *)v19 + 32LL * *(unsigned int *)(v19 + 8);
      if ( v44 == *(_QWORD *)v19 )
      {
        v20 = v40;
        v22 = 0;
      }
      else
      {
        v20 = v40;
        v21 = *(_DWORD **)v19;
        v22 = 0;
        do
        {
          v23 = *(_QWORD *)v21;
          v24 = *(unsigned int *)(*(_QWORD *)v21 + 200LL);
          if ( (_DWORD)v24 != -1 )
          {
            if ( !v21[5] )
            {
              v25 = *(_DWORD *)(*(_QWORD *)(a1 + 3944) + 16 * v24 + 12) + 1;
              if ( v22 < v25 )
                v22 = v25;
            }
            v10 = 1;
            if ( !(unsigned __int8)sub_3545640((__int64)v21, 1u) )
            {
              v10 = *(_QWORD *)(a1 + 3944) + 16LL * *(unsigned int *)(v23 + 200);
              if ( v20 > *(_DWORD *)(v10 + 4) + v21[6] * *(_DWORD *)(a1 + 3472) - v21[5] )
                v20 = *(_DWORD *)(v10 + 4) + v21[6] * *(_DWORD *)(a1 + 3472) - v21[5];
            }
          }
          v21 += 8;
        }
        while ( (_DWORD *)v44 != v21 );
      }
      i -= 4;
      v26 = 16 * v39;
      *(_DWORD *)(*(_QWORD *)(a1 + 3944) + v26 + 4) = v20;
    }
  }
  v27 = *a2;
  result = *a2 + 88LL * *((unsigned int *)a2 + 2);
  v45 = result;
  if ( result != *a2 )
  {
    do
    {
      v29 = *(__int64 **)(v27 + 32);
      result = *(unsigned int *)(v27 + 40);
      for ( j = &v29[result]; j != v29; *(_DWORD *)(v27 + 60) = result )
      {
        v31 = *v29;
        v32 = (_DWORD *)(*(_QWORD *)(a1 + 3944) + 16LL * *(unsigned int *)(*v29 + 200));
        v33 = v32[1] - *v32;
        v34 = *(unsigned int *)(v27 + 56);
        if ( v33 < (int)v34 )
          v33 = *(_DWORD *)(v27 + 56);
        *(_DWORD *)(v27 + 56) = v33;
        if ( (*(_BYTE *)(v31 + 254) & 1) == 0 )
          sub_2F8F5D0(v31, (_QWORD *)v10, v34, a4, a5, a6);
        result = *(unsigned int *)(v27 + 60);
        if ( *(_DWORD *)(v31 + 240) >= (unsigned int)result )
          result = *(unsigned int *)(v31 + 240);
        ++v29;
      }
      v27 += 88;
    }
    while ( v27 != v45 );
  }
  return result;
}
