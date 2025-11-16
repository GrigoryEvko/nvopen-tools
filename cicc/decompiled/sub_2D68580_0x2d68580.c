// Function: sub_2D68580
// Address: 0x2d68580
//
void __fastcall sub_2D68580(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r15
  _QWORD *v6; // r8
  unsigned __int64 v7; // rbx
  __int64 v8; // r9
  unsigned __int64 v9; // rsi
  _QWORD *v10; // r13
  unsigned __int64 *v11; // rbx
  _QWORD *i; // r15
  unsigned __int64 v13; // rax
  _QWORD *v14; // r13
  _QWORD *v15; // rbx
  __int64 v16; // rax
  _QWORD *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  _QWORD *v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // r13
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  _QWORD *v29; // rbx
  __int64 v30; // rax
  _QWORD *v31; // rbx
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // [rsp-50h] [rbp-50h]
  _QWORD *v36; // [rsp-50h] [rbp-50h]
  _QWORD *v37; // [rsp-48h] [rbp-48h]
  _QWORD *v38; // [rsp-48h] [rbp-48h]
  _QWORD *v39; // [rsp-48h] [rbp-48h]
  _QWORD *v40; // [rsp-48h] [rbp-48h]
  _QWORD *v41; // [rsp-48h] [rbp-48h]
  int v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v4 = (_QWORD *)(a2 + 16);
    v6 = *(_QWORD **)a1;
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v9 = *(unsigned int *)(a2 + 8);
      v42 = v9;
      if ( v9 <= v7 )
      {
        v19 = *(_QWORD *)a1;
        if ( v9 )
        {
          v31 = &v6[4 * v9];
          do
          {
            v32 = v4[2];
            v33 = v6[2];
            if ( v32 != v33 )
            {
              if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
              {
                v40 = v6;
                sub_BD60C0(v6);
                v6 = v40;
              }
              v6[2] = v32;
              if ( v32 != -4096 && v32 != 0 && v32 != -8192 )
              {
                v41 = v6;
                sub_BD73F0((__int64)v6);
                v6 = v41;
              }
            }
            v34 = v4[3];
            v6 += 4;
            v4 += 4;
            *(v6 - 1) = v34;
          }
          while ( v6 != v31 );
          v19 = *(_QWORD *)a1;
          v7 = *(unsigned int *)(a1 + 8);
        }
        v20 = (_QWORD *)(v19 + 32 * v7);
        while ( v6 != v20 )
        {
          v21 = *(v20 - 2);
          v20 -= 4;
          if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
          {
            v37 = v6;
            sub_BD60C0(v20);
            v6 = v37;
          }
        }
        *(_DWORD *)(a1 + 8) = v9;
        v22 = *(_QWORD **)a2;
        v23 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v23 )
        {
          do
          {
            v24 = *(v23 - 2);
            v23 -= 4;
            if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
              sub_BD60C0(v23);
          }
          while ( v22 != v23 );
        }
      }
      else
      {
        if ( v9 > *(unsigned int *)(a1 + 12) )
        {
          v29 = &v6[4 * v7];
          while ( v29 != (_QWORD *)v8 )
          {
            while ( 1 )
            {
              v30 = *(v29 - 2);
              v29 -= 4;
              LOBYTE(a4) = v30 != 0;
              if ( v30 == 0 || v30 == -4096 || v30 == -8192 )
                break;
              v36 = (_QWORD *)v8;
              sub_BD60C0(v29);
              v8 = (__int64)v36;
              if ( v29 == v36 )
                goto LABEL_56;
            }
          }
LABEL_56:
          *(_DWORD *)(a1 + 8) = 0;
          v7 = 0;
          sub_2D68450(a1, v9, a3, a4, (__int64)v6, v8);
          v4 = *(_QWORD **)a2;
          v9 = *(unsigned int *)(a2 + 8);
          v8 = *(_QWORD *)a1;
          v10 = *(_QWORD **)a2;
        }
        else
        {
          v10 = v4;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v7 *= 32LL;
            v25 = (_QWORD *)((char *)v6 + v7);
            do
            {
              v26 = v4[2];
              v27 = v6[2];
              if ( v26 != v27 )
              {
                if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
                {
                  v35 = v4[2];
                  v38 = v6;
                  sub_BD60C0(v6);
                  v26 = v35;
                  v6 = v38;
                }
                v6[2] = v26;
                if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
                {
                  v39 = v6;
                  sub_BD73F0((__int64)v6);
                  v6 = v39;
                }
              }
              v28 = v4[3];
              v6 += 4;
              v4 += 4;
              *(v6 - 1) = v28;
            }
            while ( v6 != v25 );
            v4 = *(_QWORD **)a2;
            v9 = *(unsigned int *)(a2 + 8);
            v8 = *(_QWORD *)a1;
            v10 = (_QWORD *)(*(_QWORD *)a2 + v7);
          }
        }
        v11 = (unsigned __int64 *)(v8 + v7);
        for ( i = &v4[4 * v9]; i != v10; v11 += 4 )
        {
          if ( v11 )
          {
            *v11 = 0;
            v11[1] = 0;
            v13 = v10[2];
            v11[2] = v13;
            if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
              sub_BD6050(v11, *v10 & 0xFFFFFFFFFFFFFFF8LL);
            v11[3] = v10[3];
          }
          v10 += 4;
        }
        *(_DWORD *)(a1 + 8) = v42;
        v14 = *(_QWORD **)a2;
        v15 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v15 )
        {
          do
          {
            v16 = *(v15 - 2);
            v15 -= 4;
            if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
              sub_BD60C0(v15);
          }
          while ( v14 != v15 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v17 = &v6[4 * v7];
      if ( v17 != v6 )
      {
        do
        {
          v18 = *(v17 - 2);
          v17 -= 4;
          if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
          {
            v43 = v8;
            sub_BD60C0(v17);
            v8 = v43;
          }
        }
        while ( v17 != (_QWORD *)v8 );
        v6 = *(_QWORD **)a1;
      }
      if ( v6 != (_QWORD *)(a1 + 16) )
        _libc_free((unsigned __int64)v6);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v4;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
