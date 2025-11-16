// Function: sub_3226DF0
// Address: 0x3226df0
//
__int64 __fastcall sub_3226DF0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  unsigned int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // di
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // r12
  _QWORD *v20; // rdi
  unsigned __int64 *v21; // r14
  unsigned __int64 *v22; // r12
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // r12
  _QWORD *v25; // [rsp+20h] [rbp-40h]
  __int64 v26; // [rsp+28h] [rbp-38h]

  v3 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL);
  v4 = sub_321F7C0(a1, a2);
  v5 = v3[218];
  v6 = (__int64)(v3 + 217);
  if ( !v5 )
    goto LABEL_8;
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(v5 + 16);
      v8 = *(_QWORD *)(v5 + 24);
      if ( v4 <= *(_DWORD *)(v5 + 32) )
        break;
      v5 = *(_QWORD *)(v5 + 24);
      if ( !v8 )
        goto LABEL_6;
    }
    v6 = v5;
    v5 = *(_QWORD *)(v5 + 16);
  }
  while ( v7 );
LABEL_6:
  if ( v3 + 217 == (_QWORD *)v6 || v4 < *(_DWORD *)(v6 + 32) )
  {
LABEL_8:
    v25 = v3 + 217;
    v9 = v6;
    v6 = sub_22077B0(0x260u);
    *(_DWORD *)(v6 + 32) = v4;
    memset((void *)(v6 + 40), 0, 0x238u);
    *(_BYTE *)(v6 + 553) = 1;
    *(_QWORD *)(v6 + 48) = v6 + 64;
    *(_QWORD *)(v6 + 56) = 0x300000000LL;
    *(_QWORD *)(v6 + 168) = 0x300000000LL;
    *(_QWORD *)(v6 + 432) = 0x1000000000LL;
    *(_QWORD *)(v6 + 472) = v6 + 488;
    *(_QWORD *)(v6 + 160) = v6 + 176;
    *(_QWORD *)(v6 + 440) = v6 + 456;
    *(_QWORD *)(v6 + 592) = v6 + 608;
    v10 = sub_E55F30(v3 + 216, v9, (unsigned int *)(v6 + 32));
    v26 = v10;
    if ( v11 )
    {
      v12 = v25 == (_QWORD *)v11 || v10 || v4 < *(_DWORD *)(v11 + 32);
      sub_220F040(v12, v6, (_QWORD *)v11, v25);
      ++v3[221];
    }
    else
    {
      sub_C7D6A0(0, 0, 8);
      v14 = *(_QWORD *)(v6 + 472);
      if ( v6 + 488 != v14 )
        j_j___libc_free_0(v14);
      v15 = *(_QWORD *)(v6 + 440);
      if ( v6 + 456 != v15 )
        j_j___libc_free_0(v15);
      v16 = *(_QWORD *)(v6 + 416);
      if ( *(_DWORD *)(v6 + 428) )
      {
        v17 = *(unsigned int *)(v6 + 424);
        if ( (_DWORD)v17 )
        {
          v18 = 8 * v17;
          v19 = 0;
          do
          {
            v20 = *(_QWORD **)(v16 + v19);
            if ( v20 != (_QWORD *)-8LL && v20 )
            {
              sub_C7D6A0((__int64)v20, *v20 + 17LL, 8);
              v16 = *(_QWORD *)(v6 + 416);
            }
            v19 += 8;
          }
          while ( v18 != v19 );
        }
      }
      _libc_free(v16);
      v21 = *(unsigned __int64 **)(v6 + 160);
      v22 = &v21[10 * *(unsigned int *)(v6 + 168)];
      if ( v21 != v22 )
      {
        do
        {
          v22 -= 10;
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
            j_j___libc_free_0(*v22);
        }
        while ( v21 != v22 );
        v22 = *(unsigned __int64 **)(v6 + 160);
      }
      if ( (unsigned __int64 *)(v6 + 176) != v22 )
        _libc_free((unsigned __int64)v22);
      v23 = *(unsigned __int64 **)(v6 + 48);
      v24 = &v23[4 * *(unsigned int *)(v6 + 56)];
      if ( v23 != v24 )
      {
        do
        {
          v24 -= 4;
          if ( (unsigned __int64 *)*v24 != v24 + 2 )
            j_j___libc_free_0(*v24);
        }
        while ( v23 != v24 );
        v24 = *(unsigned __int64 **)(v6 + 48);
      }
      if ( (unsigned __int64 *)(v6 + 64) != v24 )
        _libc_free((unsigned __int64)v24);
      j_j___libc_free_0(v6);
      v6 = v26;
    }
  }
  return sub_E784B0(v6 + 560, *(_QWORD *)(*(_QWORD *)(a2 + 472) + 16LL * *(unsigned int *)(a2 + 480) - 8));
}
