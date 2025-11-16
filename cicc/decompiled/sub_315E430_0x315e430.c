// Function: sub_315E430
// Address: 0x315e430
//
__int64 __fastcall sub_315E430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  const void *v7; // r13
  __int64 *v8; // rbx
  __int64 *v9; // r15
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v20; // [rsp+8h] [rbp-38h]

  v6 = a2;
  v7 = (const void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  v8 = *(__int64 **)(*(_QWORD *)a2 + 144LL);
  v9 = &v8[5 * *(unsigned int *)(*(_QWORD *)a2 + 152LL)];
  if ( v9 != v8 )
  {
    v10 = *v8;
    v11 = a1 + 16;
    v12 = v8 + 5;
    v13 = 0;
    while ( 1 )
    {
      *(_QWORD *)(v11 + 8 * v13) = v10;
      v13 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v13;
      if ( v9 == v12 )
        break;
      v10 = *v12;
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v20 = v6;
        sub_C8D5F0(a1, v7, v13 + 1, 8u, v6, a6);
        v13 = *(unsigned int *)(a1 + 8);
        v6 = v20;
      }
      v11 = *(_QWORD *)a1;
      v12 += 5;
    }
  }
  v14 = *(_QWORD *)(v6 + 8);
  v15 = *(__int64 **)v14;
  v16 = *(_QWORD *)v14 + 48LL * *(unsigned int *)(v14 + 8);
  if ( v16 != *(_QWORD *)v14 )
  {
    v17 = *(unsigned int *)(a1 + 8);
    do
    {
      v18 = *v15;
      if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, v7, v17 + 1, 8u, v6, a6);
        v17 = *(unsigned int *)(a1 + 8);
      }
      v15 += 6;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v17) = v18;
      v17 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v17;
    }
    while ( (__int64 *)v16 != v15 );
  }
  return a1;
}
