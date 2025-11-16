// Function: sub_2731610
// Address: 0x2731610
//
void __fastcall sub_2731610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  __int64 v9; // r15
  unsigned __int64 v10; // rdx
  int v11; // r14d
  unsigned __int64 *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // r15
  __int64 i; // r12
  __int64 v17; // rax
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r15
  __int64 v20; // rbx
  unsigned __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rbx
  __int64 v25; // rsi
  unsigned __int64 v26; // [rsp-48h] [rbp-48h]
  unsigned __int64 v27; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(unsigned int *)(a2 + 8);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = v8;
    v12 = *(unsigned __int64 **)a1;
    if ( v8 <= v10 )
    {
      v17 = *(_QWORD *)a1;
      if ( v8 )
      {
        v20 = *(_QWORD *)a2;
        v21 = v9 + 160 * v8;
        do
        {
          v22 = v20;
          v23 = v9;
          v9 += 160;
          v20 += 160;
          sub_272D7C0(v23, v22, v10, a4, a5, a6);
          *(_QWORD *)(v9 - 16) = *(_QWORD *)(v20 - 16);
          *(_QWORD *)(v9 - 8) = *(_QWORD *)(v20 - 8);
        }
        while ( v9 != v21 );
        v17 = *(_QWORD *)a1;
        v10 = *(unsigned int *)(a1 + 8);
      }
      v18 = (unsigned __int64 *)(v17 + 160 * v10);
      while ( (unsigned __int64 *)v9 != v18 )
      {
        v18 -= 20;
        if ( (unsigned __int64 *)*v18 != v18 + 2 )
          _libc_free(*v18);
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        v19 = &v12[20 * v10];
        while ( v19 != v12 )
        {
          while ( 1 )
          {
            v19 -= 20;
            if ( (unsigned __int64 *)*v19 == v19 + 2 )
              break;
            _libc_free(*v19);
            if ( v19 == v12 )
              goto LABEL_22;
          }
        }
LABEL_22:
        *(_DWORD *)(a1 + 8) = 0;
        sub_2366E20(a1, v8, v10, a4, a5, a6);
        v8 = *(unsigned int *)(a2 + 8);
        v12 = *(unsigned __int64 **)a1;
        v10 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v24 = *(_QWORD *)a2;
        v10 *= 160LL;
        v26 = v9 + v10;
        do
        {
          v25 = v24;
          v27 = v10;
          v24 += 160;
          sub_272D7C0(v9, v25, v10, a4, a5, a6);
          v9 += 160;
          v10 = v27;
          *(_QWORD *)(v9 - 16) = *(_QWORD *)(v24 - 16);
          a4 = *(_QWORD *)(v24 - 8);
          *(_QWORD *)(v9 - 8) = a4;
        }
        while ( v9 != v26 );
        v8 = *(unsigned int *)(a2 + 8);
        v12 = *(unsigned __int64 **)a1;
      }
      v13 = *(_QWORD *)a2;
      v14 = (__int64)v12 + v10;
      v15 = v13 + 160 * v8;
      for ( i = v10 + v13; v15 != i; v14 += 160 )
      {
        if ( v14 )
        {
          *(_DWORD *)(v14 + 8) = 0;
          *(_QWORD *)v14 = v14 + 16;
          *(_DWORD *)(v14 + 12) = 8;
          if ( *(_DWORD *)(i + 8) )
            sub_272D7C0(v14, i, v10, a4, a5, a6);
          *(_QWORD *)(v14 + 144) = *(_QWORD *)(i + 144);
          *(_QWORD *)(v14 + 152) = *(_QWORD *)(i + 152);
        }
        i += 160;
      }
    }
    *(_DWORD *)(a1 + 8) = v11;
  }
}
