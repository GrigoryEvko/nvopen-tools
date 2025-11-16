// Function: sub_2DDD530
// Address: 0x2ddd530
//
void __fastcall sub_2DDD530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 *v9; // rbx
  unsigned __int64 v10; // r8
  unsigned __int64 *v11; // r13
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rbx
  unsigned __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned __int64 *v20; // r13
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r13
  __int64 v23; // rdx
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r13
  unsigned __int64 *v26; // rbx
  unsigned __int64 *v27; // rbx
  __int64 v28; // rdi
  unsigned __int64 *v29; // r15
  unsigned __int64 *v30; // rbx
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // [rsp-50h] [rbp-50h]
  int v34; // [rsp-44h] [rbp-44h]
  unsigned __int64 v35; // [rsp-40h] [rbp-40h]
  unsigned __int64 v36; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v9 = *(unsigned __int64 **)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v34 = v12;
      if ( v12 <= v10 )
      {
        v23 = *(_QWORD *)a1;
        if ( v12 )
        {
          v30 = &v11[6 * v12];
          do
          {
            v31 = v6;
            v32 = (__int64)v11;
            v11 += 6;
            v6 += 48;
            sub_2DDB710(v32, v31, v23, a4, v10, a6);
          }
          while ( v11 != v30 );
          v23 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(a1 + 8);
        }
        v24 = (unsigned __int64 *)(v23 + 48 * v10);
        while ( v11 != v24 )
        {
          v24 -= 6;
          if ( (unsigned __int64 *)*v24 != v24 + 2 )
            _libc_free(*v24);
        }
        *(_DWORD *)(a1 + 8) = v34;
        v25 = *(unsigned __int64 **)a2;
        v26 = (unsigned __int64 *)(*(_QWORD *)a2 + 48LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v26 )
        {
          do
          {
            v26 -= 6;
            if ( (unsigned __int64 *)*v26 != v26 + 2 )
              _libc_free(*v26);
          }
          while ( v25 != v26 );
        }
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 12);
        if ( v12 > v13 )
        {
          v29 = &v9[6 * v10];
          while ( v9 != v29 )
          {
            while ( 1 )
            {
              v29 -= 6;
              if ( (unsigned __int64 *)*v29 == v29 + 2 )
                break;
              _libc_free(*v29);
              if ( v9 == v29 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          sub_2DDD420(a1, v12, v13, a4, v10, a6);
          v6 = *(_QWORD *)a2;
          v9 = *(unsigned __int64 **)a1;
          v10 = 0;
          v12 = *(unsigned int *)(a2 + 8);
          v14 = *(_QWORD *)a2;
        }
        else
        {
          v14 = v6;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v33 = 48 * v10;
            v10 *= 48LL;
            v27 = (unsigned __int64 *)((char *)v9 + v10);
            do
            {
              v28 = (__int64)v11;
              v11 += 6;
              v36 = v10;
              sub_2DDB710(v28, v6, v13, v14, v10, a6);
              v6 += 48;
              v10 = v36;
            }
            while ( v27 != v11 );
            v6 = *(_QWORD *)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v9 = *(unsigned __int64 **)a1;
            v14 = *(_QWORD *)a2 + v33;
          }
        }
        v15 = (__int64)v9 + v10;
        v16 = v6 + 48 * v12;
        v17 = v14;
        if ( v16 != v14 )
        {
          do
          {
            while ( 1 )
            {
              if ( v15 )
              {
                *(_DWORD *)(v15 + 8) = 0;
                *(_QWORD *)v15 = v15 + 16;
                *(_DWORD *)(v15 + 12) = 4;
                v18 = *(unsigned int *)(v17 + 8);
                if ( (_DWORD)v18 )
                  break;
              }
              v17 += 48;
              v15 += 48;
              if ( v16 == v17 )
                goto LABEL_12;
            }
            v19 = v17;
            v35 = v16;
            v17 += 48;
            sub_2DDB710(v15, v19, v18, v14, v10, a6);
            v16 = v35;
            v15 += 48;
          }
          while ( v35 != v17 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v34;
        v20 = *(unsigned __int64 **)a2;
        v21 = (unsigned __int64 *)(*(_QWORD *)a2 + 48LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v21 )
        {
          do
          {
            v21 -= 6;
            if ( (unsigned __int64 *)*v21 != v21 + 2 )
              _libc_free(*v21);
          }
          while ( v20 != v21 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v22 = &v9[6 * v10];
      if ( v9 != v22 )
      {
        do
        {
          v22 -= 6;
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
            _libc_free(*v22);
        }
        while ( v9 != v22 );
        v22 = *(unsigned __int64 **)a1;
      }
      if ( v22 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v22);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
