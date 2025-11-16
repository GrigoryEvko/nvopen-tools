// Function: sub_359B100
// Address: 0x359b100
//
__int64 __fastcall sub_359B100(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 **v7; // rbx
  _QWORD *v8; // r10
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  const void *v13; // rsi
  __int64 v14; // r15
  __int64 **v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 v18; // r12
  __int64 v19; // r15
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // [rsp+0h] [rbp-50h]
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  __int64 **v29; // [rsp+10h] [rbp-40h]
  _QWORD *v30; // [rsp+18h] [rbp-38h]
  _QWORD *v31; // [rsp+18h] [rbp-38h]

  v7 = (__int64 **)a1[5];
  v8 = (_QWORD *)a1[7];
  v27 = a1[6];
  v9 = (__int64 *)a1[2];
  v29 = (__int64 **)a1[9];
  if ( v7 == v29 )
  {
    v24 = (v27 - (__int64)v9) >> 3;
    if ( v27 - (__int64)v9 > 0 )
    {
      v25 = *(unsigned int *)(a2 + 8);
      do
      {
        v26 = *v9;
        if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v25 + 1, 8u, a5, a6);
          v25 = *(unsigned int *)(a2 + 8);
        }
        ++v9;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v25) = v26;
        v25 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v25;
        --v24;
      }
      while ( v24 );
    }
  }
  else
  {
    v10 = a1[4] - (_QWORD)v9;
    v11 = v10 >> 3;
    if ( v10 > 0 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = (const void *)(a2 + 16);
      do
      {
        v14 = *v9;
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v28 = v8;
          sub_C8D5F0(a2, v13, v12 + 1, 8u, a5, a6);
          v12 = *(unsigned int *)(a2 + 8);
          v8 = v28;
        }
        ++v9;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v12) = v14;
        v12 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v12;
        --v11;
      }
      while ( v11 );
    }
    v15 = v7 + 1;
    if ( v29 != v15 )
    {
      v16 = *(unsigned int *)(a2 + 8);
      do
      {
        v17 = *v15;
        v18 = (__int64)(*v15 + 64);
        do
        {
          v19 = *v17;
          if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            v30 = v8;
            sub_C8D5F0(a2, (const void *)(a2 + 16), v16 + 1, 8u, a5, a6);
            v16 = *(unsigned int *)(a2 + 8);
            v8 = v30;
          }
          ++v17;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v16) = v19;
          v16 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
          *(_DWORD *)(a2 + 8) = v16;
        }
        while ( v17 != (__int64 *)v18 );
        ++v15;
      }
      while ( v29 != v15 );
    }
    v20 = (v27 - (__int64)v8) >> 3;
    if ( v27 - (__int64)v8 > 0 )
    {
      v21 = *(unsigned int *)(a2 + 8);
      do
      {
        v22 = *v8;
        if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v31 = v8;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v21 + 1, 8u, a5, a6);
          v21 = *(unsigned int *)(a2 + 8);
          v8 = v31;
        }
        ++v8;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v21) = v22;
        v21 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v21;
        --v20;
      }
      while ( v20 );
    }
  }
  return a2;
}
