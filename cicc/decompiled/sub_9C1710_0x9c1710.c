// Function: sub_9C1710
// Address: 0x9c1710
//
__int64 __fastcall sub_9C1710(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rsi
  __int64 v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // r14
  _QWORD *v7; // r12
  __int64 v8; // rdx
  unsigned int v9; // r15d
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 i; // r9
  _QWORD *v13; // r15
  _QWORD *v14; // r13
  __int64 v15; // rax
  _QWORD *v16; // r14
  _QWORD *v17; // r15
  int v18; // r13d
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+28h] [rbp-48h]
  __int64 v24[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = (_QWORD *)(a1 + 16);
  v4 = sub_C8D7D0(a1, a1 + 16, a2, 80, v24);
  v5 = *(_QWORD **)a1;
  v22 = v4;
  v6 = v4;
  v7 = (_QWORD *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 == v7 )
    goto LABEL_26;
  do
  {
    while ( 1 )
    {
      if ( !v6 )
        goto LABEL_3;
      v8 = v6 + 16;
      *(_DWORD *)(v6 + 8) = 0;
      *(_QWORD *)v6 = v6 + 16;
      *(_DWORD *)(v6 + 12) = 1;
      v9 = *((_DWORD *)v5 + 2);
      if ( !v9 || v5 == (_QWORD *)v6 )
        goto LABEL_3;
      v10 = *v5;
      v3 = v5 + 2;
      if ( v5 + 2 == (_QWORD *)*v5 )
        break;
      *(_QWORD *)v6 = v10;
      *(_DWORD *)(v6 + 8) = *((_DWORD *)v5 + 2);
      *(_DWORD *)(v6 + 12) = *((_DWORD *)v5 + 3);
      *v5 = v3;
      *((_DWORD *)v5 + 3) = 0;
      *((_DWORD *)v5 + 2) = 0;
LABEL_3:
      v5 += 10;
      v6 += 80;
      if ( v7 == v5 )
        goto LABEL_16;
    }
    v11 = 1;
    if ( v9 != 1 )
    {
      sub_9C1610(v6, v9);
      v8 = *(_QWORD *)v6;
      v10 = *v5;
      v11 = *((unsigned int *)v5 + 2);
    }
    for ( i = v10 + (v11 << 6); i != v10; v8 += 64 )
    {
      if ( v8 )
      {
        *(_DWORD *)(v8 + 8) = 0;
        *(_QWORD *)v8 = v8 + 16;
        *(_DWORD *)(v8 + 12) = 12;
        if ( *(_DWORD *)(v10 + 8) )
        {
          v20 = i;
          v21 = v10;
          v23 = v8;
          sub_9B68C0(v8, (char **)v10);
          i = v20;
          v10 = v21;
          v8 = v23;
        }
      }
      v10 += 64;
    }
    *(_DWORD *)(v6 + 8) = v9;
    v3 = (_QWORD *)*v5;
    v13 = (_QWORD *)(*v5 + ((unsigned __int64)*((unsigned int *)v5 + 2) << 6));
    if ( (_QWORD *)*v5 != v13 )
    {
      do
      {
        v13 -= 8;
        if ( (_QWORD *)*v13 != v13 + 2 )
          _libc_free(*v13, v3);
      }
      while ( v3 != v13 );
    }
    *((_DWORD *)v5 + 2) = 0;
    v5 += 10;
    v6 += 80;
  }
  while ( v7 != v5 );
LABEL_16:
  v14 = *(_QWORD **)a1;
  v7 = (_QWORD *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v7 )
  {
    do
    {
      v15 = *((unsigned int *)v7 - 18);
      v16 = (_QWORD *)*(v7 - 10);
      v7 -= 10;
      v15 <<= 6;
      v17 = (_QWORD *)((char *)v16 + v15);
      if ( v16 != (_QWORD *)((char *)v16 + v15) )
      {
        do
        {
          v17 -= 8;
          if ( (_QWORD *)*v17 != v17 + 2 )
            _libc_free(*v17, v3);
        }
        while ( v16 != v17 );
        v16 = (_QWORD *)*v7;
      }
      if ( v16 != v7 + 2 )
        _libc_free(v16, v3);
    }
    while ( v7 != v14 );
    v7 = *(_QWORD **)a1;
  }
LABEL_26:
  v18 = v24[0];
  if ( (_QWORD *)(a1 + 16) != v7 )
    _libc_free(v7, v3);
  *(_DWORD *)(a1 + 12) = v18;
  *(_QWORD *)a1 = v22;
  return v22;
}
