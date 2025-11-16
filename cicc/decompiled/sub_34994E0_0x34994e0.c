// Function: sub_34994E0
// Address: 0x34994e0
//
void __fastcall sub_34994E0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // r8
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // rdx
  int v7; // r15d
  unsigned __int64 *v8; // rax
  unsigned __int64 *v9; // r12
  __int64 *v10; // rbx
  unsigned __int64 *v11; // r13
  unsigned __int64 *i; // r12
  unsigned __int64 *v13; // r12
  unsigned __int64 *v14; // rbx
  unsigned __int64 *v15; // rbx
  unsigned __int64 *v16; // r12
  unsigned __int64 *v17; // rsi
  unsigned __int64 *v18; // rdi
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // rsi
  unsigned __int64 *v21; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v22; // [rsp-48h] [rbp-48h]
  unsigned __int64 v23; // [rsp-40h] [rbp-40h]
  unsigned __int64 v24; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v4 = *(unsigned int *)(a2 + 8);
    v5 = *(unsigned __int64 **)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v7 = *(_DWORD *)(a2 + 8);
    v8 = *(unsigned __int64 **)a1;
    if ( v4 <= v6 )
    {
      if ( *(_DWORD *)(a2 + 8) )
      {
        v15 = *(unsigned __int64 **)a2;
        v16 = &v5[4 * v4];
        do
        {
          v17 = v15;
          v18 = v5;
          v5 += 4;
          v15 += 4;
          sub_2240AE0(v18, v17);
        }
        while ( v5 != v16 );
        v8 = *(unsigned __int64 **)a1;
        v6 = *(unsigned int *)(a1 + 8);
      }
      v13 = &v8[4 * v6];
      while ( v5 != v13 )
      {
        v13 -= 4;
        if ( (unsigned __int64 *)*v13 != v13 + 2 )
          j_j___libc_free_0(*v13);
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
            if ( (unsigned __int64 *)*v14 == v14 + 2 )
              break;
            v21 = v8;
            v23 = v4;
            j_j___libc_free_0(*v14);
            v8 = v21;
            v4 = v23;
            if ( v14 == v21 )
              goto LABEL_20;
          }
        }
LABEL_20:
        *(_DWORD *)(a1 + 8) = 0;
        sub_95D880(a1, v4);
        v4 = *(unsigned int *)(a2 + 8);
        v8 = *(unsigned __int64 **)a1;
        v6 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v6 *= 32LL;
        v19 = *(unsigned __int64 **)a2;
        v22 = (unsigned __int64 *)((char *)v5 + v6);
        do
        {
          v20 = v19;
          v24 = v6;
          v19 += 4;
          sub_2240AE0(v5, v20);
          v5 += 4;
          v6 = v24;
        }
        while ( v22 != v5 );
        v4 = *(unsigned int *)(a2 + 8);
        v8 = *(unsigned __int64 **)a1;
      }
      v9 = *(unsigned __int64 **)a2;
      v10 = (__int64 *)((char *)v8 + v6);
      v11 = &v9[4 * v4];
      for ( i = (unsigned __int64 *)((char *)v9 + v6); v11 != i; v10 += 4 )
      {
        if ( v10 )
        {
          *v10 = (__int64)(v10 + 2);
          sub_343F8D0(v10, (_BYTE *)*i, *i + i[1]);
        }
        i += 4;
      }
    }
    *(_DWORD *)(a1 + 8) = v7;
  }
}
