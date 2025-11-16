// Function: sub_EA88D0
// Address: 0xea88d0
//
unsigned __int64 __fastcall sub_EA88D0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r15
  _QWORD *v12; // rbx
  __int64 v13; // r15
  _QWORD *v14; // rdi
  _QWORD *v15; // rbx
  __int64 v16; // r14
  __int64 *v17; // r15
  __int64 v19; // rsi
  unsigned __int64 v20; // rbx
  __int64 *v21; // r15
  _QWORD *v22; // rbx
  _QWORD *v23; // r14
  __int64 v24; // [rsp+0h] [rbp-50h]
  unsigned __int64 v25; // [rsp+8h] [rbp-48h]
  unsigned __int64 v26; // [rsp+8h] [rbp-48h]
  unsigned __int64 v27[7]; // [rsp+18h] [rbp-38h] BYREF

  result = *(unsigned int *)(a1 + 12);
  if ( result < a2 )
  {
    v19 = a1 + 16;
    v20 = a2;
    v24 = sub_C8D7D0(a1, a1 + 16, a2, 0x20u, v27, a6);
    v21 = (__int64 *)v24;
    do
    {
      if ( v21 )
      {
        *v21 = (__int64)(v21 + 2);
        v19 = *a3;
        sub_EA2980(v21, (_BYTE *)*a3, *a3 + a3[1]);
      }
      v21 += 4;
      --v20;
    }
    while ( v20 );
    v22 = *(_QWORD **)a1;
    v23 = (_QWORD *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    if ( *(_QWORD **)a1 != v23 )
    {
      do
      {
        v23 -= 4;
        if ( (_QWORD *)*v23 != v23 + 2 )
        {
          v19 = v23[2] + 1LL;
          j_j___libc_free_0(*v23, v19);
        }
      }
      while ( v22 != v23 );
      v23 = *(_QWORD **)a1;
    }
    result = v27[0];
    if ( (_QWORD *)(a1 + 16) != v23 )
    {
      v26 = v27[0];
      _libc_free(v23, v19);
      result = v26;
    }
    *(_DWORD *)(a1 + 8) = a2;
    *(_DWORD *)(a1 + 12) = result;
    *(_QWORD *)a1 = v24;
  }
  else
  {
    v10 = *(unsigned int *)(a1 + 8);
    v11 = a2;
    if ( v10 <= a2 )
      v11 = *(unsigned int *)(a1 + 8);
    if ( v11 )
    {
      v12 = *(_QWORD **)a1;
      v13 = *(_QWORD *)a1 + 32 * v11;
      do
      {
        v14 = v12;
        v12 += 4;
        result = sub_2240AE0(v14, a3);
      }
      while ( (_QWORD *)v13 != v12 );
      v10 = *(unsigned int *)(a1 + 8);
    }
    if ( v10 < a2 )
    {
      v17 = (__int64 *)(*(_QWORD *)a1 + 32 * v10);
      result = a2 - v10;
      if ( a2 != v10 )
      {
        do
        {
          if ( v17 )
          {
            v25 = result;
            *v17 = (__int64)(v17 + 2);
            sub_EA2980(v17, (_BYTE *)*a3, *a3 + a3[1]);
            result = v25;
          }
          v17 += 4;
          --result;
        }
        while ( result );
      }
    }
    else if ( v10 > a2 )
    {
      v15 = (_QWORD *)(*(_QWORD *)a1 + 32 * v10);
      v16 = *(_QWORD *)a1 + 32 * a2;
      while ( (_QWORD *)v16 != v15 )
      {
        while ( 1 )
        {
          v15 -= 4;
          result = (unsigned __int64)(v15 + 2);
          if ( (_QWORD *)*v15 == v15 + 2 )
            break;
          result = j_j___libc_free_0(*v15, v15[2] + 1LL);
          if ( (_QWORD *)v16 == v15 )
            goto LABEL_14;
        }
      }
    }
LABEL_14:
    *(_DWORD *)(a1 + 8) = a2;
  }
  return result;
}
