// Function: sub_35BFB40
// Address: 0x35bfb40
//
__int64 __fastcall sub_35BFB40(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 v3; // rcx
  unsigned __int64 v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // r12
  __int64 v7; // r10
  __int64 v8; // rsi
  _DWORD *v9; // rcx
  unsigned int v10; // r11d
  unsigned int v11; // r14d
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned int v14; // ecx
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rsi
  int v18; // esi
  __int64 v19; // r9
  unsigned int v20; // ecx
  __int64 v21; // rsi
  __int64 v22; // r9
  unsigned int v23; // ecx
  __int64 v24; // rsi
  __int64 v25; // r8
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 result; // rax
  _QWORD *v29; // r12
  volatile signed __int32 *v30; // rdi
  volatile signed __int32 *v31; // rbx
  __int64 i; // [rsp+0h] [rbp-60h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h] BYREF
  volatile signed __int32 *v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v37; // [rsp+28h] [rbp-38h]

  v3 = *a3;
  v4 = a3[1];
  *a3 = 0;
  a3[1] = 0;
  v36 = v3;
  v37 = v4;
  sub_35BF7B0(&v34, a1 + 120, &v36);
  if ( v37 )
    j_j___libc_free_0_0(v37);
  v5 = *(_QWORD **)(a1 + 152);
  v6 = 48LL * a2;
  if ( v5 )
  {
    v7 = v34;
    v8 = *(_QWORD *)(*v5 + 160LL);
    v9 = (_DWORD *)(v6 + *(_QWORD *)(*v5 + 208LL));
    v10 = v9[5];
    v11 = v9[6];
    v12 = *(_QWORD *)v9;
    v13 = v8 + 96LL * v10;
    *(_DWORD *)(v13 + 24) -= *(_DWORD *)(*(_QWORD *)v9 + 20LL);
    v14 = 0;
    v15 = *(_QWORD *)(v12 + 24);
    v33 = v13 + 16;
    v16 = v8 + 96LL * v11;
    for ( i = v16 + 16;
          *(_DWORD *)(v13 + 20) > v14;
          *(_DWORD *)(*(_QWORD *)(v13 + 32) + 4 * v17) -= *(unsigned __int8 *)(v15 + v17) )
    {
      v17 = v14++;
    }
    v18 = *(_DWORD *)(v16 + 20);
    *(_DWORD *)(v16 + 24) -= *(_DWORD *)(v12 + 16);
    v19 = *(_QWORD *)(v12 + 32);
    if ( v18 )
    {
      v20 = 0;
      do
      {
        v21 = v20++;
        *(_DWORD *)(*(_QWORD *)(v16 + 32) + 4 * v21) -= *(unsigned __int8 *)(v19 + v21);
      }
      while ( *(_DWORD *)(v16 + 20) > v20 );
    }
    *(_DWORD *)(v13 + 24) += *(_DWORD *)(v7 + 20);
    v22 = *(_QWORD *)(v7 + 24);
    if ( *(_DWORD *)(v13 + 20) )
    {
      v23 = 0;
      do
      {
        v24 = v23++;
        *(_DWORD *)(*(_QWORD *)(v13 + 32) + 4 * v24) += *(unsigned __int8 *)(v22 + v24);
      }
      while ( *(_DWORD *)(v13 + 20) > v23 );
    }
    *(_DWORD *)(v16 + 24) += *(_DWORD *)(v7 + 16);
    v25 = *(_QWORD *)(v7 + 32);
    if ( *(_DWORD *)(v16 + 20) )
    {
      v26 = 0;
      do
      {
        v27 = v26++;
        *(_DWORD *)(*(_QWORD *)(v16 + 32) + 4 * v27) += *(unsigned __int8 *)(v25 + v27);
      }
      while ( *(_DWORD *)(v16 + 20) > v26 );
    }
    sub_35BA930(v5, v10, v33);
    sub_35BA930(v5, v11, i);
  }
  result = v34;
  v29 = (_QWORD *)(*(_QWORD *)(a1 + 208) + v6);
  v30 = (volatile signed __int32 *)v29[1];
  *v29 = v34;
  v31 = v35;
  if ( v35 != v30 )
  {
    if ( v35 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v35 + 2, 1u);
      else
        ++*((_DWORD *)v35 + 2);
      v30 = (volatile signed __int32 *)v29[1];
    }
    if ( v30 )
      result = sub_A191D0(v30);
    v29[1] = v31;
    v30 = v35;
  }
  if ( v30 )
    return sub_A191D0(v30);
  return result;
}
