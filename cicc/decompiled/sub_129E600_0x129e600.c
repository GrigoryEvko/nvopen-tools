// Function: sub_129E600
// Address: 0x129e600
//
__int64 __fastcall sub_129E600(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r12
  _QWORD *v8; // rdi
  __int64 *v9; // r14
  __int64 **v10; // r13
  __int64 *v11; // r12
  __int64 *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 *v15; // rbx
  unsigned __int64 v16; // r12
  __int64 v17; // rdi
  __int64 i; // [rsp+8h] [rbp-58h]
  __int64 *v20; // [rsp+10h] [rbp-50h]
  unsigned __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 *v23; // [rsp+28h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 632);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 616);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        a2 = v3[1];
        if ( a2 )
          sub_161E7C0(v3 + 1);
      }
      v3 += 2;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 616));
  v5 = *(unsigned int *)(a1 + 600);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD **)(a1 + 584);
    v7 = &v6[2 * v5];
    do
    {
      while ( 1 )
      {
        if ( *v6 != -2 && *v6 != -1 )
        {
          a2 = v6[1];
          if ( a2 )
            break;
        }
        v6 += 2;
        if ( v7 == v6 )
          goto LABEL_14;
      }
      v8 = v6 + 1;
      v6 += 2;
      sub_161E7C0(v8);
    }
    while ( v7 != v6 );
  }
LABEL_14:
  j___libc_free_0(*(_QWORD *)(a1 + 584));
  v9 = *(__int64 **)(a1 + 552);
  v23 = *(__int64 **)(a1 + 544);
  v10 = (__int64 **)(*(_QWORD *)(a1 + 536) + 8LL);
  v22 = *(_QWORD *)(a1 + 568);
  v11 = *(__int64 **)(a1 + 512);
  v20 = *(__int64 **)(a1 + 528);
  for ( i = *(_QWORD *)(a1 + 536); v22 > (unsigned __int64)v10; ++v10 )
  {
    v12 = *v10;
    v13 = (__int64)(*v10 + 64);
    do
    {
      a2 = *v12;
      if ( *v12 )
        sub_161E7C0(v12);
      ++v12;
    }
    while ( (__int64 *)v13 != v12 );
  }
  if ( v22 == i )
  {
    while ( v23 != v11 )
    {
      a2 = *v11;
      if ( *v11 )
        sub_161E7C0(v11);
      ++v11;
    }
  }
  else
  {
    for ( ; v20 != v11; ++v11 )
    {
      a2 = *v11;
      if ( *v11 )
        sub_161E7C0(v11);
    }
    for ( ; v23 != v9; ++v9 )
    {
      a2 = *v9;
      if ( *v9 )
        sub_161E7C0(v9);
    }
  }
  v14 = *(_QWORD *)(a1 + 496);
  if ( v14 )
  {
    v15 = *(__int64 **)(a1 + 536);
    v16 = *(_QWORD *)(a1 + 568) + 8LL;
    if ( v16 > (unsigned __int64)v15 )
    {
      do
      {
        v17 = *v15++;
        j_j___libc_free_0(v17, 512);
      }
      while ( v16 > (unsigned __int64)v15 );
      v14 = *(_QWORD *)(a1 + 496);
    }
    a2 = 8LL * *(_QWORD *)(a1 + 504);
    j_j___libc_free_0(v14, a2);
  }
  return sub_129E320(a1 + 16, a2);
}
