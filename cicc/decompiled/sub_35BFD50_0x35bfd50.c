// Function: sub_35BFD50
// Address: 0x35bfd50
//
__int64 __fastcall sub_35BFD50(_QWORD *a1, int a2, int a3, __int64 *a4)
{
  __int64 v5; // rdx
  volatile signed __int32 *v6; // rax
  volatile signed __int32 *v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // r12d
  __int64 v10; // r10
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rsi
  int v15; // eax
  __int64 v16; // r8
  unsigned int i; // eax
  __int64 v18; // rdx
  unsigned int *v19; // r9
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v26; // [rsp+0h] [rbp-60h] BYREF
  volatile signed __int32 *v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h] BYREF
  volatile signed __int32 *v29; // [rsp+18h] [rbp-48h]
  int v30; // [rsp+24h] [rbp-3Ch]
  int v31; // [rsp+28h] [rbp-38h]
  __int64 v32; // [rsp+30h] [rbp-30h]
  __int64 v33; // [rsp+38h] [rbp-28h]

  v5 = *a4;
  v6 = (volatile signed __int32 *)a4[1];
  *a4 = 0;
  a4[1] = 0;
  v28 = v5;
  v29 = v6;
  sub_35BF7B0(&v26, (__int64)(a1 + 15), &v28);
  if ( v29 )
    j_j___libc_free_0_0((unsigned __int64)v29);
  v7 = v27;
  v8 = v26;
  if ( v27 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd(v27 + 2, 1u);
    else
      ++*((_DWORD *)v27 + 2);
  }
  v31 = a3;
  v28 = v8;
  v29 = v7;
  v30 = a2;
  v32 = -1;
  v33 = -1;
  v9 = sub_35BCFD0(a1, (__int64)&v28);
  if ( v29 )
    sub_A191D0(v29);
  v10 = a1[19];
  if ( v10 )
  {
    v11 = 48LL * v9;
    v12 = v11 + *(_QWORD *)(*(_QWORD *)v10 + 208LL);
    v13 = *(_QWORD *)v12;
    v14 = *(_QWORD *)(*(_QWORD *)v10 + 160LL) + 96LL * *(unsigned int *)(v12 + 20);
    v15 = *(_DWORD *)(v14 + 24);
    if ( *(_DWORD *)(v12 + 20) == *(_DWORD *)(v12 + 24) )
    {
      *(_DWORD *)(v14 + 24) = *(_DWORD *)(v13 + 16) + v15;
      v16 = *(_QWORD *)(v13 + 32);
    }
    else
    {
      *(_DWORD *)(v14 + 24) = *(_DWORD *)(v13 + 20) + v15;
      v16 = *(_QWORD *)(v13 + 24);
    }
    for ( i = 0; *(_DWORD *)(v14 + 20) > i; *(_DWORD *)(*(_QWORD *)(v14 + 32) + 4 * v18) += *(unsigned __int8 *)(v16 + v18) )
      v18 = i++;
    v19 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)v10 + 208LL) + v11);
    v20 = *(_QWORD *)(*(_QWORD *)v10 + 160LL) + 96LL * v19[6];
    v21 = *(_QWORD *)v19;
    *(_DWORD *)(v20 + 24) += *(_DWORD *)(*(_QWORD *)v19 + 16LL);
    v22 = *(_QWORD *)(v21 + 32);
    if ( *(_DWORD *)(v20 + 20) )
    {
      v23 = 0;
      do
      {
        v24 = v23++;
        *(_DWORD *)(*(_QWORD *)(v20 + 32) + 4 * v24) += *(unsigned __int8 *)(v22 + v24);
      }
      while ( *(_DWORD *)(v20 + 20) > v23 );
    }
  }
  if ( v27 )
    sub_A191D0(v27);
  return v9;
}
