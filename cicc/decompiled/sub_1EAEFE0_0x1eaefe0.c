// Function: sub_1EAEFE0
// Address: 0x1eaefe0
//
__int64 __fastcall sub_1EAEFE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        unsigned __int64 *a5,
        unsigned int *a6,
        unsigned int a7)
{
  __int64 v7; // r14
  unsigned int *v8; // rbx
  __int64 result; // rax
  unsigned __int64 *v11; // r12
  unsigned int v12; // ecx
  __int64 v13; // r9
  unsigned int v14; // eax
  unsigned int v15; // edx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  __int64 v18; // rdi
  unsigned int *v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // r13
  _BOOL4 v22; // r10d
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned int v26; // eax
  int *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r15
  _BOOL4 v31; // r14d
  unsigned __int64 *v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  unsigned int *v34; // [rsp+18h] [rbp-78h]
  __int64 v35; // [rsp+20h] [rbp-70h]
  _BOOL4 v36; // [rsp+28h] [rbp-68h]
  __int64 v37; // [rsp+28h] [rbp-68h]
  unsigned int *v39; // [rsp+40h] [rbp-50h]
  unsigned int v41[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v7 = a2;
  v8 = *(unsigned int **)(a1 + 48);
  result = (__int64)&v8[*(unsigned int *)(a1 + 56)];
  v35 = a2 + 80;
  v39 = (unsigned int *)result;
  if ( (unsigned int *)result != v8 )
  {
    v11 = a5;
    do
    {
      v12 = *v8;
      v41[0] = *v8;
      if ( a4 )
        *v11 += *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL * (*(_DWORD *)(a3 + 32) + v12) + 8);
      v13 = *(_QWORD *)(a3 + 8);
      v14 = *(_DWORD *)(v13 + 40LL * (*(_DWORD *)(a3 + 32) + v12) + 16);
      v15 = v14;
      if ( *a6 >= v14 )
        v15 = *a6;
      *a6 = v15;
      v16 = v14 * ((v14 + *v11 - 1 - a7 % (unsigned __int64)v14) / v14) + a7 % (unsigned __int64)v14;
      *v11 = v16;
      if ( a4 )
      {
        a5 = *(unsigned __int64 **)(a3 + 8);
        a5[5 * *(_DWORD *)(a3 + 32) + 5 * v12] = -(__int64)v16;
        if ( *(_QWORD *)(v7 + 120) )
          goto LABEL_17;
      }
      else
      {
        *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL * (*(_DWORD *)(a3 + 32) + v12)) = v16;
        *v11 = *(_QWORD *)(*(_QWORD *)(a3 + 8) + 40LL * (*(_DWORD *)(a3 + 32) + v12) + 8) + v16;
        if ( *(_QWORD *)(v7 + 120) )
          goto LABEL_17;
      }
      v17 = *(unsigned int *)(v7 + 8);
      v18 = *(_QWORD *)v7;
      v19 = (unsigned int *)(*(_QWORD *)v7 + 4 * v17);
      if ( *(unsigned int **)v7 != v19 )
      {
        result = *(_QWORD *)v7;
        while ( *(_DWORD *)result != v41[0] )
        {
          result += 4;
          if ( v19 == (unsigned int *)result )
            goto LABEL_20;
        }
        if ( (unsigned int *)result != v19 )
          goto LABEL_14;
      }
LABEL_20:
      v37 = v7 + 88;
      if ( v17 <= 0xF )
      {
        if ( *(_DWORD *)(v7 + 8) >= *(_DWORD *)(v7 + 12) )
        {
          sub_16CD150(v7, (const void *)(v7 + 16), 0, 4, (int)a5, v13);
          v19 = (unsigned int *)(*(_QWORD *)v7 + 4LL * *(unsigned int *)(v7 + 8));
        }
        result = v41[0];
        *v19 = v41[0];
        ++*(_DWORD *)(v7 + 8);
        goto LABEL_14;
      }
      v32 = v11;
      v24 = v7;
      v34 = v8;
      v33 = a3;
      while ( 1 )
      {
        v27 = (int *)(v18 + 4 * v17 - 4);
        v28 = sub_BB80C0(v35, v27);
        v30 = v29;
        if ( v29 )
        {
          v31 = 1;
          if ( !v28 && v29 != v37 )
            v31 = *v27 < *(_DWORD *)(v29 + 32);
          v25 = sub_22077B0(40);
          *(_DWORD *)(v25 + 32) = *v27;
          sub_220F040(v31, v25, v30, v37);
          ++*(_QWORD *)(v24 + 120);
        }
        v26 = *(_DWORD *)(v24 + 8) - 1;
        *(_DWORD *)(v24 + 8) = v26;
        if ( !v26 )
          break;
        v18 = *(_QWORD *)v24;
        v17 = v26;
      }
      v7 = v24;
      v8 = v34;
      a3 = v33;
      v11 = v32;
LABEL_17:
      result = sub_BB80C0(v35, (int *)v41);
      v21 = v20;
      if ( v20 )
      {
        v22 = 1;
        if ( !result && v20 != v7 + 88 )
          v22 = (signed int)v41[0] < *(_DWORD *)(v20 + 32);
        v36 = v22;
        v23 = sub_22077B0(40);
        *(_DWORD *)(v23 + 32) = v41[0];
        result = sub_220F040(v36, v23, v21, v7 + 88);
        ++*(_QWORD *)(v7 + 120);
      }
LABEL_14:
      ++v8;
    }
    while ( v8 != v39 );
  }
  return result;
}
