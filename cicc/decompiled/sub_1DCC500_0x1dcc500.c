// Function: sub_1DCC500
// Address: 0x1dcc500
//
unsigned int *__fastcall sub_1DCC500(unsigned int *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // r13
  _QWORD *i; // r14
  _QWORD *v8; // rbx
  _QWORD *v9; // r15
  __int64 v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // rax
  _QWORD *v14; // rbx
  __int64 v15; // rdi
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  _QWORD *v18; // rdi
  unsigned int v20; // [rsp+0h] [rbp-60h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  unsigned __int64 v23; // [rsp+18h] [rbp-48h]
  int v24; // [rsp+24h] [rbp-3Ch]
  __int64 v25; // [rsp+28h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v3 = (((((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
         | (a1[3] + 2LL)
         | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4)
       | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 8)
     | (((((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
       | (a1[3] + 2LL)
       | (((unsigned __int64)a1[3] + 2) >> 1)) >> 4)
     | (((a1[3] + 2LL) | (((unsigned __int64)a1[3] + 2) >> 1)) >> 2)
     | (a1[3] + 2LL)
     | (((unsigned __int64)a1[3] + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  v5 = 0xFFFFFFFFLL;
  if ( v4 >= a2 )
    v2 = v4;
  if ( v2 <= 0xFFFFFFFF )
    v5 = v2;
  v20 = v5;
  v21 = malloc(56 * v5);
  if ( !v21 )
    sub_16BD1C0("Allocation failed", 1u);
  v23 = *(_QWORD *)a1 + 56LL * a1[2];
  if ( *(_QWORD *)a1 != v23 )
  {
    v6 = (_QWORD *)v21;
    for ( i = (_QWORD *)(*(_QWORD *)a1 + 8LL); ; i += 7 )
    {
      if ( v6 )
      {
        v8 = v6 + 1;
        *v6 = 0;
        v6[2] = v6 + 1;
        v6[1] = v6 + 1;
        v6[3] = 0;
        v9 = (_QWORD *)*i;
        if ( (_QWORD *)*i != i )
        {
          do
          {
            v10 = v9[4];
            v24 = *((_DWORD *)v9 + 4);
            v25 = v9[3];
            v11 = sub_22077B0(40);
            *(_QWORD *)(v11 + 32) = v10;
            *(_DWORD *)(v11 + 16) = v24;
            *(_QWORD *)(v11 + 24) = v25;
            sub_2208C80(v11, v6 + 1);
            ++v6[3];
            v9 = (_QWORD *)*v9;
          }
          while ( v9 != i );
          v8 = (_QWORD *)v6[1];
        }
        *v6 = v8;
        v6[4] = i[3];
        v6[5] = i[4];
        v6[6] = i[5];
        i[5] = 0;
        i[4] = 0;
        i[3] = 0;
      }
      v6 += 7;
      if ( (_QWORD *)v23 == i + 6 )
        break;
    }
    v12 = *(_QWORD **)a1;
    v13 = *(_QWORD *)a1 + 56LL * a1[2];
    v23 = v13;
    if ( *(_QWORD *)a1 != v13 )
    {
      v14 = (_QWORD *)(v13 - 48);
      do
      {
        v15 = v14[3];
        v16 = v14 - 1;
        if ( v15 )
          j_j___libc_free_0(v15, v14[5] - v15);
        v17 = (_QWORD *)*v14;
        while ( v17 != v14 )
        {
          v18 = v17;
          v17 = (_QWORD *)*v17;
          j_j___libc_free_0(v18, 40);
        }
        v14 -= 7;
      }
      while ( v16 != v12 );
      v23 = *(_QWORD *)a1;
    }
  }
  if ( (unsigned int *)v23 != a1 + 4 )
    _libc_free(v23);
  *(_QWORD *)a1 = v21;
  a1[3] = v20;
  return a1;
}
