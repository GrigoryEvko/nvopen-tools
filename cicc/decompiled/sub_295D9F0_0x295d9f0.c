// Function: sub_295D9F0
// Address: 0x295d9f0
//
void __fastcall sub_295D9F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  __int64 v9; // rcx
  unsigned __int64 v10; // rbx
  _QWORD *v11; // r8
  unsigned __int64 v12; // r12
  _QWORD *v13; // rsi
  __int64 v14; // rax
  unsigned __int64 *v15; // rbx
  _QWORD *v16; // r12
  _QWORD *v17; // rdx
  unsigned __int64 v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdi
  _QWORD *v25; // rbx
  _QWORD *v26; // r12
  _QWORD *v27; // rbx
  __int64 v28; // rcx
  __int64 v29; // rsi
  _QWORD *v30; // rbx
  __int64 v31; // rax
  _QWORD *v32; // r12
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // [rsp-58h] [rbp-58h]
  __int64 v37; // [rsp-58h] [rbp-58h]
  __int64 v38; // [rsp-50h] [rbp-50h]
  __int64 v39; // [rsp-50h] [rbp-50h]
  int v40; // [rsp-44h] [rbp-44h]
  _QWORD *v41; // [rsp-40h] [rbp-40h]
  __int64 v42; // [rsp-40h] [rbp-40h]
  _QWORD *v43; // [rsp-40h] [rbp-40h]
  _QWORD *v44; // [rsp-40h] [rbp-40h]
  _QWORD *v45; // [rsp-40h] [rbp-40h]
  _QWORD *v46; // [rsp-40h] [rbp-40h]
  __int64 v47; // [rsp-40h] [rbp-40h]
  __int64 v48; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = (_QWORD *)(a2 + 16);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v40 = *(_DWORD *)(a2 + 8);
      if ( v12 <= v10 )
      {
        v24 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v31 = 24 * v12;
          v32 = (_QWORD *)v9;
          v37 = v31;
          v33 = a2 + v31 + 16;
          do
          {
            v34 = v6[2];
            v35 = v32[2];
            if ( v34 != v35 )
            {
              if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
              {
                v39 = v9;
                v47 = v6[2];
                sub_BD60C0(v32);
                v9 = v39;
                v34 = v47;
              }
              v32[2] = v34;
              if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
              {
                v48 = v9;
                sub_BD73F0((__int64)v32);
                v9 = v48;
              }
            }
            v6 += 3;
            v32 += 3;
          }
          while ( v6 != (_QWORD *)v33 );
          v24 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(a1 + 8);
          v11 = (_QWORD *)(v9 + v37);
        }
        v25 = (_QWORD *)(v24 + 24 * v10);
        if ( v25 != v11 )
        {
          do
          {
            v25 -= 3;
            v43 = v11;
            sub_D68D70(v25);
            v11 = v43;
          }
          while ( v43 != v25 );
        }
        *(_DWORD *)(a1 + 8) = v40;
        v26 = *(_QWORD **)a2;
        v27 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v27 )
        {
          do
          {
            v27 -= 3;
            sub_D68D70(v27);
          }
          while ( v26 != v27 );
        }
      }
      else
      {
        if ( v12 > *(unsigned int *)(a1 + 12) )
        {
          v30 = (_QWORD *)(v9 + 24 * v10);
          if ( v30 != (_QWORD *)v9 )
          {
            do
            {
              v30 -= 3;
              v46 = (_QWORD *)v9;
              sub_D68D70(v30);
              v9 = (__int64)v46;
            }
            while ( v30 != v46 );
          }
          *(_DWORD *)(a1 + 8) = 0;
          v10 = 0;
          sub_D6B530(a1, v12, a3, v9, (__int64)v11, a6);
          v6 = *(_QWORD **)a2;
          v12 = *(unsigned int *)(a2 + 8);
          v9 = *(_QWORD *)a1;
          v13 = *(_QWORD **)a2;
        }
        else
        {
          v13 = (_QWORD *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v36 = 24 * v10;
            v10 *= 24LL;
            do
            {
              v28 = v6[2];
              v29 = v11[2];
              if ( v28 != v29 )
              {
                if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
                {
                  v38 = v6[2];
                  v44 = v11;
                  sub_BD60C0(v11);
                  v28 = v38;
                  v11 = v44;
                }
                v11[2] = v28;
                if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
                {
                  v45 = v11;
                  sub_BD73F0((__int64)v11);
                  v11 = v45;
                }
              }
              v6 += 3;
              v11 += 3;
            }
            while ( v6 != (_QWORD *)(a2 + v10 + 16) );
            v6 = *(_QWORD **)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v9 = *(_QWORD *)a1;
            v13 = (_QWORD *)(*(_QWORD *)a2 + v36);
          }
        }
        v14 = 3 * v12;
        v15 = (unsigned __int64 *)(v9 + v10);
        v16 = v13;
        v17 = &v6[v14];
        if ( v17 != v13 )
        {
          do
          {
            if ( v15 )
            {
              *v15 = 0;
              v15[1] = 0;
              v18 = v16[2];
              v15[2] = v18;
              if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
              {
                v41 = v17;
                sub_BD6050(v15, *v16 & 0xFFFFFFFFFFFFFFF8LL);
                v17 = v41;
              }
            }
            v16 += 3;
            v15 += 3;
          }
          while ( v17 != v16 );
        }
        *(_DWORD *)(a1 + 8) = v40;
        v19 = *(_QWORD **)a2;
        v20 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v20 )
        {
          do
          {
            v21 = *(v20 - 1);
            v20 -= 3;
            if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
              sub_BD60C0(v20);
          }
          while ( v19 != v20 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v22 = (_QWORD *)(v9 + 24 * v10);
      if ( v22 != (_QWORD *)v9 )
      {
        do
        {
          v23 = *(v22 - 1);
          v22 -= 3;
          if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
          {
            v42 = v9;
            sub_BD60C0(v22);
            v9 = v42;
          }
        }
        while ( v22 != (_QWORD *)v9 );
        v11 = *(_QWORD **)a1;
      }
      if ( v11 != (_QWORD *)(a1 + 16) )
        _libc_free((unsigned __int64)v11);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
