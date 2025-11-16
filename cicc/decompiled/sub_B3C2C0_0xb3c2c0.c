// Function: sub_B3C2C0
// Address: 0xb3c2c0
//
void __fastcall sub_B3C2C0(__int64 a1, __int64 *a2)
{
  unsigned __int64 v4; // r8
  _QWORD *v5; // r13
  unsigned __int64 v6; // rdx
  int v7; // r15d
  _QWORD *v8; // rax
  __int64 v9; // r12
  __int64 *v10; // rbx
  __int64 v11; // r13
  __int64 i; // r12
  _QWORD *v13; // r12
  _QWORD *v14; // rbx
  __int64 v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // rsi
  _QWORD *v18; // rdi
  __int64 v19; // rbx
  __int64 v20; // rsi
  _QWORD *v21; // [rsp-48h] [rbp-48h]
  _QWORD *v22; // [rsp-48h] [rbp-48h]
  unsigned __int64 v23; // [rsp-40h] [rbp-40h]
  unsigned __int64 v24; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v4 = *((unsigned int *)a2 + 2);
    v5 = *(_QWORD **)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v7 = *((_DWORD *)a2 + 2);
    v8 = *(_QWORD **)a1;
    if ( v4 <= v6 )
    {
      if ( *((_DWORD *)a2 + 2) )
      {
        v15 = *a2;
        v16 = &v5[4 * v4];
        do
        {
          v17 = v15;
          v18 = v5;
          v5 += 4;
          v15 += 32;
          sub_2240AE0(v18, v17);
        }
        while ( v5 != v16 );
        v8 = *(_QWORD **)a1;
        v6 = *(unsigned int *)(a1 + 8);
      }
      v13 = &v8[4 * v6];
      while ( v5 != v13 )
      {
        v13 -= 4;
        if ( (_QWORD *)*v13 != v13 + 2 )
          j_j___libc_free_0(*v13, v13[2] + 1LL);
      }
    }
    else
    {
      if ( v4 > *(unsigned int *)(a1 + 12) )
      {
        v14 = &v5[4 * v6];
        while ( v14 != v8 )
        {
          while ( 1 )
          {
            v14 -= 4;
            if ( (_QWORD *)*v14 == v14 + 2 )
              break;
            v21 = v8;
            v23 = v4;
            j_j___libc_free_0(*v14, v14[2] + 1LL);
            v8 = v21;
            v4 = v23;
            if ( v14 == v21 )
              goto LABEL_20;
          }
        }
LABEL_20:
        *(_DWORD *)(a1 + 8) = 0;
        sub_95D880(a1, v4);
        v4 = *((unsigned int *)a2 + 2);
        v8 = *(_QWORD **)a1;
        v6 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v6 *= 32LL;
        v19 = *a2;
        v22 = (_QWORD *)((char *)v5 + v6);
        do
        {
          v20 = v19;
          v24 = v6;
          v19 += 32;
          sub_2240AE0(v5, v20);
          v5 += 4;
          v6 = v24;
        }
        while ( v22 != v5 );
        v4 = *((unsigned int *)a2 + 2);
        v8 = *(_QWORD **)a1;
      }
      v9 = *a2;
      v10 = (_QWORD *)((char *)v8 + v6);
      v11 = v9 + 32 * v4;
      for ( i = v6 + v9; v11 != i; v10 += 4 )
      {
        if ( v10 )
        {
          *v10 = (__int64)(v10 + 2);
          sub_B3AE60(v10, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
        }
        i += 32;
      }
    }
    *(_DWORD *)(a1 + 8) = v7;
  }
}
