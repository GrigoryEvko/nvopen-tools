// Function: sub_391AD00
// Address: 0x391ad00
//
__int64 __fastcall sub_391AD00(__int64 a1, int *a2, _QWORD *a3)
{
  int v5; // ecx
  __int64 v7; // r15
  int v8; // esi
  int *v9; // r9
  __int64 v10; // r12
  int v11; // r11d
  int *v12; // rax
  int *v13; // rdx
  int v14; // r8d
  int *v15; // r10
  __int64 v16; // r13
  int *v17; // rdx
  int *v18; // r8
  int v19; // ebx
  int v20; // ecx
  int *v21; // rbx
  int v22; // r14d
  int v23; // ebx
  _DWORD *v24; // r8
  int *v25; // rsi
  __int64 result; // rax
  _DWORD *v27; // r8
  int *v28; // rsi
  __int64 *v29; // [rsp-50h] [rbp-50h]
  __int64 v30; // [rsp-48h] [rbp-48h]
  __int64 v31; // [rsp-40h] [rbp-40h] BYREF
  __int64 v32; // [rsp-8h] [rbp-8h] BYREF

  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    *a3 = 0;
    return 0;
  }
  *(&v32 - 12) = (__int64)(&v32 - 10);
  *(&v32 - 19) = 0x100000000LL;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *a2;
  *(&v32 - 11) = 0x100000000LL;
  *(&v32 - 20) = (__int64)(&v32 - 18);
  v9 = (int *)*((_QWORD *)a2 + 1);
  v10 = (unsigned int)a2[4];
  *(&v32 - 17) = (__int64)(&v32 - 15);
  v11 = v8;
  *(&v32 - 16) = 0x400000000LL;
  v30 = 0x400000000LL;
  v12 = &v9[v10];
  *((_DWORD *)&v32 - 42) = 1;
  v29 = &v31;
  *((_DWORD *)&v32 - 26) = 2;
  if ( v9 != v12 )
  {
    v13 = v9;
    do
    {
      v14 = *v13++;
      v8 += 37 * v14;
    }
    while ( v12 != v13 );
  }
  v15 = (int *)*((_QWORD *)a2 + 4);
  v16 = (unsigned int)a2[10];
  v17 = &v15[v16];
  if ( v15 != v17 )
  {
    v18 = (int *)*((_QWORD *)a2 + 4);
    do
    {
      v19 = *v18++;
      v8 += 37 * v19;
    }
    while ( v17 != v18 );
  }
  *((_DWORD *)&v32 - 44) = 1;
  v20 = v5 - 1;
  *(&v32 - 23) = 0;
  *((_DWORD *)&v32 - 43) = v20 & v8;
  while ( 1 )
  {
    v21 = (int *)(v7 + 72LL * *((unsigned int *)&v32 - 43));
    v22 = *v21;
    if ( v11 == *v21 && v10 == v21[4] )
    {
      v24 = (_DWORD *)*((_QWORD *)v21 + 1);
      if ( v9 != v12 )
      {
        *(&v32 - 24) = (__int64)v17;
        v25 = v9;
        do
        {
          if ( *v25 != *v24 )
          {
            v17 = (int *)*(&v32 - 24);
            goto LABEL_10;
          }
          ++v25;
          ++v24;
        }
        while ( v12 != v25 );
        v17 = (int *)*(&v32 - 24);
      }
      if ( v16 == v21[10] )
      {
        v27 = (_DWORD *)*((_QWORD *)v21 + 4);
        if ( v15 == v17 )
        {
LABEL_38:
          *a3 = v21;
          return 1;
        }
        *(&v32 - 24) = (__int64)v12;
        v28 = v15;
        while ( *v28 == *v27 )
        {
          ++v28;
          ++v27;
          if ( v17 == v28 )
            goto LABEL_38;
        }
        v12 = (int *)*(&v32 - 24);
      }
    }
LABEL_10:
    if ( v22 == 1 )
      break;
    if ( v22 == 2 && !v21[4] && !v21[10] )
    {
      if ( *(&v32 - 23) )
        v21 = (int *)*(&v32 - 23);
      *(&v32 - 23) = (__int64)v21;
    }
LABEL_17:
    v23 = *((_DWORD *)&v32 - 44);
    *((_DWORD *)&v32 - 43) = v20 & (v23 + *((_DWORD *)&v32 - 43));
    *((_DWORD *)&v32 - 44) = v23 + 1;
  }
  if ( v21[4] || v21[10] )
    goto LABEL_17;
  if ( *(&v32 - 23) )
    v21 = (int *)*(&v32 - 23);
  result = 0;
  *a3 = v21;
  return result;
}
