// Function: sub_2D29780
// Address: 0x2d29780
//
void __fastcall sub_2D29780(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 v8; // r13
  __int64 v9; // r15
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // r13
  _QWORD *v13; // r14
  unsigned __int8 *v14; // rsi
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rax
  unsigned __int64 v21; // r13
  __int64 v22; // rsi
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rsi
  _QWORD *v26; // r15
  _QWORD *v27; // r13
  _QWORD *v28; // r14
  unsigned __int8 *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rsi
  unsigned int v33; // r14d
  _QWORD *v34; // r13
  _QWORD *v35; // r14
  unsigned __int8 *v36; // rsi
  __int64 v37; // rax
  __int64 v38; // [rsp-60h] [rbp-60h]
  __int64 v39; // [rsp-58h] [rbp-58h]
  unsigned __int64 v40; // [rsp-58h] [rbp-58h]
  __int64 v41; // [rsp-58h] [rbp-58h]
  unsigned int v42; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v43; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != (unsigned int *)a2 )
  {
    v6 = a2 + 16;
    v8 = a1[2];
    v9 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v42 = *(_DWORD *)(a2 + 8);
      if ( v10 <= v8 )
      {
        v20 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v34 = (_QWORD *)(v9 + 16);
          v35 = (_QWORD *)(a2 + 32);
          v38 = 32 * v10;
          v41 = v9 + 16 + 32 * v10;
          do
          {
            *((_DWORD *)v34 - 4) = *((_DWORD *)v35 - 4);
            *(v34 - 1) = *(v35 - 1);
            if ( v34 != v35 )
            {
              sub_9C6650(v34);
              v36 = (unsigned __int8 *)*v35;
              *v34 = *v35;
              if ( v36 )
              {
                sub_B976B0((__int64)v35, v36, (__int64)v34);
                *v35 = 0;
              }
            }
            v37 = v35[1];
            v34 += 4;
            v35 += 4;
            *(v34 - 3) = v37;
          }
          while ( (_QWORD *)v41 != v34 );
          v20 = *(_QWORD *)a1;
          v8 = a1[2];
          v9 += v38;
        }
        v21 = v20 + 32 * v8;
        while ( v9 != v21 )
        {
          v22 = *(_QWORD *)(v21 - 16);
          v21 -= 32LL;
          if ( v22 )
            sub_B91220(v21 + 16, v22);
        }
        a1[2] = v42;
        v23 = *(_QWORD *)a2;
        v24 = *(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v24 )
        {
          do
          {
            v25 = *(_QWORD *)(v24 - 16);
            v24 -= 32;
            if ( v25 )
              sub_B91220(v24 + 16, v25);
          }
          while ( v23 != v24 );
        }
      }
      else
      {
        if ( v10 > a1[3] )
        {
          v31 = v9 + 32 * v8;
          while ( v31 != v9 )
          {
            while ( 1 )
            {
              v32 = *(_QWORD *)(v31 - 16);
              v31 -= 32;
              if ( !v32 )
                break;
              v40 = v10;
              sub_B91220(v31 + 16, v32);
              v10 = v40;
              if ( v31 == v9 )
                goto LABEL_48;
            }
          }
LABEL_48:
          a1[2] = 0;
          v9 = sub_C8D7D0((__int64)a1, (__int64)(a1 + 4), v10, 0x20u, &v43, a6);
          sub_2D296B0(a1, v9);
          v33 = v43;
          if ( a1 + 4 != *(unsigned int **)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = v9;
          a1[3] = v33;
          v6 = *(_QWORD *)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v11 = *(_QWORD *)a2;
        }
        else
        {
          v11 = a2 + 16;
          if ( a1[2] )
          {
            v26 = (_QWORD *)(v9 + 16);
            v27 = (_QWORD *)(a2 + 32);
            v39 = 32LL * a1[2];
            v28 = &v26[(unsigned __int64)v39 / 8];
            do
            {
              *((_DWORD *)v26 - 4) = *((_DWORD *)v27 - 4);
              *(v26 - 1) = *(v27 - 1);
              if ( v27 != v26 )
              {
                sub_9C6650(v26);
                v29 = (unsigned __int8 *)*v27;
                *v26 = *v27;
                if ( v29 )
                {
                  sub_B976B0((__int64)v27, v29, (__int64)v26);
                  *v27 = 0;
                }
              }
              v30 = v27[1];
              v26 += 4;
              v27 += 4;
              *(v26 - 3) = v30;
            }
            while ( v26 != v28 );
            v6 = *(_QWORD *)a2;
            v10 = *(unsigned int *)(a2 + 8);
            v9 = v39 + *(_QWORD *)a1;
            v11 = *(_QWORD *)a2 + v39;
          }
        }
        v12 = (_QWORD *)(v6 + 32 * v10);
        v13 = (_QWORD *)(v11 + 16);
        if ( v12 != (_QWORD *)v11 )
        {
          while ( 1 )
          {
            if ( v9 )
            {
              *(_DWORD *)v9 = *((_DWORD *)v13 - 4);
              *(_QWORD *)(v9 + 8) = *(v13 - 1);
              v14 = (unsigned __int8 *)*v13;
              *(_QWORD *)(v9 + 16) = *v13;
              if ( v14 )
              {
                sub_B976B0((__int64)v13, v14, v9 + 16);
                *v13 = 0;
              }
              *(_QWORD *)(v9 + 24) = v13[1];
            }
            v9 += 32;
            if ( v12 == v13 + 2 )
              break;
            v13 += 4;
          }
        }
        a1[2] = v42;
        v15 = *(_QWORD *)a2;
        v16 = *(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v16 )
        {
          do
          {
            v17 = *(_QWORD *)(v16 - 16);
            v16 -= 32;
            if ( v17 )
              sub_B91220(v16 + 16, v17);
          }
          while ( v15 != v16 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v18 = v9 + 32 * v8;
      if ( v18 != v9 )
      {
        do
        {
          v19 = *(_QWORD *)(v18 - 16);
          v18 -= 32;
          if ( v19 )
            sub_B91220(v18 + 16, v19);
        }
        while ( v18 != v9 );
        v9 = *(_QWORD *)a1;
      }
      if ( (unsigned int *)v9 != a1 + 4 )
        _libc_free(v9);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      a1[2] = *(_DWORD *)(a2 + 8);
      a1[3] = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
