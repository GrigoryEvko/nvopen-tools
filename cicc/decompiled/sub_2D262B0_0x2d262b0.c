// Function: sub_2D262B0
// Address: 0x2d262b0
//
void __fastcall sub_2D262B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r10
  __int64 v11; // rax
  unsigned __int64 v12; // r14
  _QWORD *v13; // rbx
  _QWORD *v14; // r15
  unsigned __int8 *v15; // rsi
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // rsi
  unsigned __int64 v19; // rbx
  __int64 v20; // rsi
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // r13
  __int64 v25; // rbx
  __int64 v26; // rsi
  unsigned __int8 **v27; // r8
  unsigned __int8 **v28; // rbx
  unsigned __int8 **v29; // r15
  unsigned __int8 *v30; // rsi
  unsigned __int64 v31; // rbx
  __int64 v32; // rsi
  _QWORD *v33; // r14
  _QWORD *v34; // rbx
  unsigned __int8 *v35; // rsi
  __int64 v36; // [rsp-50h] [rbp-50h]
  __int64 v37; // [rsp-50h] [rbp-50h]
  int v38; // [rsp-44h] [rbp-44h]
  __int64 v39; // [rsp-40h] [rbp-40h]
  unsigned __int64 v40; // [rsp-40h] [rbp-40h]
  unsigned __int8 **v41; // [rsp-40h] [rbp-40h]
  unsigned __int64 v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v7 = *(_QWORD *)a1;
    v8 = *(unsigned int *)(a1 + 8);
    v39 = a2 + 16;
    v9 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v38 = *(_DWORD *)(a2 + 8);
      if ( v10 <= v8 )
      {
        v21 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v33 = (_QWORD *)(a2 + 32);
          v34 = (_QWORD *)(v7 + 16);
          v37 = 24 * v10;
          v43 = v7 + 16 + 24 * v10;
          do
          {
            *((_DWORD *)v34 - 4) = *((_DWORD *)v33 - 4);
            *((_DWORD *)v34 - 3) = *((_DWORD *)v33 - 3);
            *((_DWORD *)v34 - 2) = *((_DWORD *)v33 - 2);
            *((_DWORD *)v34 - 1) = *((_DWORD *)v33 - 1);
            if ( v34 != v33 )
            {
              sub_9C6650(v34);
              v35 = (unsigned __int8 *)*v33;
              *v34 = *v33;
              if ( v35 )
              {
                sub_B976B0((__int64)v33, v35, (__int64)v34);
                *v33 = 0;
              }
            }
            v33 += 3;
            v34 += 3;
          }
          while ( (_QWORD *)v43 != v34 );
          v21 = *(_QWORD *)a1;
          v8 = *(unsigned int *)(a1 + 8);
          v9 = v7 + v37;
        }
        v22 = v21 + 24 * v8;
        while ( v9 != v22 )
        {
          v23 = *(_QWORD *)(v22 - 8);
          v22 -= 24LL;
          if ( v23 )
          {
            v40 = v9;
            sub_B91220(v22 + 16, v23);
            v9 = v40;
          }
        }
        *(_DWORD *)(a1 + 8) = v38;
        v24 = *(_QWORD *)a2;
        v25 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v25 )
        {
          do
          {
            v26 = *(_QWORD *)(v25 - 8);
            v25 -= 24;
            if ( v26 )
              sub_B91220(v25 + 16, v26);
          }
          while ( v24 != v25 );
        }
      }
      else
      {
        if ( v10 > *(unsigned int *)(a1 + 12) )
        {
          v31 = v7 + 24 * v8;
          while ( v31 != v7 )
          {
            while ( 1 )
            {
              v32 = *(_QWORD *)(v31 - 8);
              v31 -= 24LL;
              if ( !v32 )
                break;
              v42 = v10;
              sub_B91220(v31 + 16, v32);
              v10 = v42;
              if ( v31 == v7 )
                goto LABEL_47;
            }
          }
LABEL_47:
          *(_DWORD *)(a1 + 8) = 0;
          v8 = 0;
          sub_2D24250(a1, v10, a3, a4, v9, a6);
          v11 = *(_QWORD *)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v7 = *(_QWORD *)a1;
          v39 = *(_QWORD *)a2;
        }
        else
        {
          v11 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v27 = (unsigned __int8 **)(a2 + 32);
            v28 = (unsigned __int8 **)(v7 + 16);
            v36 = 24 * v8;
            v8 *= 24LL;
            v29 = (unsigned __int8 **)(v7 + 16 + v8);
            do
            {
              *((_DWORD *)v28 - 4) = *((_DWORD *)v27 - 4);
              *((_DWORD *)v28 - 3) = *((_DWORD *)v27 - 3);
              *((_DWORD *)v28 - 2) = *((_DWORD *)v27 - 2);
              *((_DWORD *)v28 - 1) = *((_DWORD *)v27 - 1);
              if ( v27 != v28 )
              {
                v41 = v27;
                sub_9C6650(v28);
                v27 = v41;
                v30 = *v41;
                *v28 = *v41;
                if ( v30 )
                {
                  sub_B976B0((__int64)v41, v30, (__int64)v28);
                  v27 = v41;
                  *v41 = 0;
                }
              }
              v28 += 3;
              v27 += 3;
            }
            while ( v28 != v29 );
            v10 = *(unsigned int *)(a2 + 8);
            v7 = *(_QWORD *)a1;
            v39 = *(_QWORD *)a2;
            v11 = *(_QWORD *)a2 + v36;
          }
        }
        v12 = v7 + v8;
        v13 = (_QWORD *)(v11 + 16);
        v14 = (_QWORD *)(v39 + 24 * v10);
        if ( v14 != (_QWORD *)v11 )
        {
          while ( 1 )
          {
            if ( v12 )
            {
              *(_DWORD *)v12 = *((_DWORD *)v13 - 4);
              *(_DWORD *)(v12 + 4) = *((_DWORD *)v13 - 3);
              *(_DWORD *)(v12 + 8) = *((_DWORD *)v13 - 2);
              *(_DWORD *)(v12 + 12) = *((_DWORD *)v13 - 1);
              v15 = (unsigned __int8 *)*v13;
              *(_QWORD *)(v12 + 16) = *v13;
              if ( v15 )
              {
                sub_B976B0((__int64)v13, v15, v12 + 16);
                *v13 = 0;
              }
            }
            v12 += 24LL;
            if ( v14 == v13 + 1 )
              break;
            v13 += 3;
          }
        }
        *(_DWORD *)(a1 + 8) = v38;
        v16 = *(_QWORD *)a2;
        v17 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v17 - 8);
            v17 -= 24;
            if ( v18 )
              sub_B91220(v17 + 16, v18);
          }
          while ( v16 != v17 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v19 = v7 + 24 * v8;
      if ( v19 != v7 )
      {
        do
        {
          v20 = *(_QWORD *)(v19 - 8);
          v19 -= 24LL;
          if ( v20 )
            sub_B91220(v19 + 16, v20);
        }
        while ( v19 != v7 );
        v9 = *(_QWORD *)a1;
      }
      if ( v9 != a1 + 16 )
        _libc_free(v9);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v39;
    }
  }
}
