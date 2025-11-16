// Function: sub_20E9F70
// Address: 0x20e9f70
//
__int64 __fastcall sub_20E9F70(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 *v5; // rbx
  __int64 *v6; // r14
  unsigned int v7; // r12d
  __int64 v8; // rax
  int v9; // r9d
  unsigned __int64 *v10; // r12
  unsigned __int64 *v11; // rbx
  unsigned __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 j; // rcx
  int v16; // r12d
  int v17; // r13d
  __int64 v18; // rax
  unsigned int v19; // r8d
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 i; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+0h] [rbp-40h]
  unsigned int v28; // [rsp+0h] [rbp-40h]

  v2 = a1 + 240;
  *(_QWORD *)(a1 + 232) = a2;
  *(_DWORD *)(a1 + 248) = 0;
  *(_DWORD *)(a1 + 288) = 0;
  sub_3945AE0(a1 + 240, 2 * (unsigned int)((__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3));
  v3 = *(_QWORD *)(a1 + 232);
  v4 = *(_QWORD *)(v3 + 328);
  for ( i = v3 + 320; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    v5 = *(__int64 **)(v4 + 96);
    v6 = *(__int64 **)(v4 + 88);
    v7 = 2 * *(_DWORD *)(v4 + 48) + 1;
    while ( v5 != v6 )
    {
      v8 = *v6++;
      sub_3945B70(v2, v7, (unsigned int)(2 * *(_DWORD *)(v8 + 48)));
    }
  }
  sub_3945BD0(v2);
  if ( byte_4FCF820 )
    sub_20E9F00((const char *)a1);
  v10 = *(unsigned __int64 **)(a1 + 296);
  v11 = &v10[6 * *(unsigned int *)(a1 + 304)];
  while ( v10 != v11 )
  {
    while ( 1 )
    {
      v11 -= 6;
      if ( (unsigned __int64 *)*v11 == v11 + 2 )
        break;
      _libc_free(*v11);
      if ( v10 == v11 )
        goto LABEL_11;
    }
  }
LABEL_11:
  *(_DWORD *)(a1 + 304) = 0;
  v12 = *(unsigned int *)(a1 + 288);
  if ( *(_DWORD *)(a1 + 288) )
  {
    v13 = 0;
    if ( v12 > *(unsigned int *)(a1 + 308) )
    {
      sub_1E1E640(a1 + 296, v12);
      v13 = 48LL * *(unsigned int *)(a1 + 304);
    }
    v27 = *(_QWORD *)(a1 + 296);
    v14 = v27 + v13;
    for ( j = v27 + 48 * v12; j != v14; v14 += 48 )
    {
      if ( v14 )
      {
        *(_DWORD *)(v14 + 8) = 0;
        *(_QWORD *)v14 = v14 + 16;
        *(_DWORD *)(v14 + 12) = 8;
      }
    }
    *(_DWORD *)(a1 + 304) = v12;
  }
  v16 = 0;
  v17 = (__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 104LL) - *(_QWORD *)(*(_QWORD *)(a1 + 232) + 96LL)) >> 3;
  if ( v17 )
  {
    do
    {
      v18 = *(_QWORD *)(a1 + 240);
      v19 = *(_DWORD *)(v18 + 4LL * (unsigned int)(2 * v16));
      v20 = *(unsigned int *)(v18 + 4LL * (unsigned int)(2 * v16 + 1));
      v21 = *(_QWORD *)(a1 + 296) + 48LL * v19;
      v22 = *(unsigned int *)(v21 + 8);
      if ( (unsigned int)v22 >= *(_DWORD *)(v21 + 12) )
      {
        v28 = v19;
        sub_16CD150(v21, (const void *)(v21 + 16), 0, 4, v19, v9);
        v22 = *(unsigned int *)(v21 + 8);
        v19 = v28;
      }
      *(_DWORD *)(*(_QWORD *)v21 + 4 * v22) = v16;
      ++*(_DWORD *)(v21 + 8);
      if ( (_DWORD)v20 != v19 )
      {
        v23 = *(_QWORD *)(a1 + 296) + 48 * v20;
        v24 = *(unsigned int *)(v23 + 8);
        if ( (unsigned int)v24 >= *(_DWORD *)(v23 + 12) )
        {
          sub_16CD150(*(_QWORD *)(a1 + 296) + 48 * v20, (const void *)(v23 + 16), 0, 4, v19, v9);
          v24 = *(unsigned int *)(v23 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v23 + 4 * v24) = v16;
        ++*(_DWORD *)(v23 + 8);
      }
      ++v16;
    }
    while ( v16 != v17 );
  }
  return 0;
}
