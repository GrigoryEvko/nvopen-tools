// Function: sub_2844B50
// Address: 0x2844b50
//
void __fastcall sub_2844B50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  _QWORD *v9; // rsi
  unsigned __int64 v10; // rbx
  _QWORD *v11; // r8
  unsigned __int64 v12; // r15
  _QWORD *v13; // r12
  unsigned __int64 *v14; // rbx
  _QWORD *i; // r15
  unsigned __int64 v16; // rax
  _QWORD *v17; // r12
  _QWORD *v18; // rbx
  __int64 v19; // rax
  _QWORD *v20; // r12
  __int64 v21; // rax
  _QWORD *v22; // rax
  _QWORD *v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // r12
  _QWORD *v26; // rbx
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rsi
  _QWORD *v30; // rbx
  __int64 v31; // rax
  _QWORD *v32; // r12
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // rdi
  __int64 v36; // [rsp-58h] [rbp-58h]
  __int64 v37; // [rsp-58h] [rbp-58h]
  __int64 v38; // [rsp-50h] [rbp-50h]
  __int64 v39; // [rsp-50h] [rbp-50h]
  _QWORD *v40; // [rsp-48h] [rbp-48h]
  _QWORD *v41; // [rsp-48h] [rbp-48h]
  _QWORD *v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-48h] [rbp-48h]
  __int64 v44; // [rsp-48h] [rbp-48h]
  int v45; // [rsp-40h] [rbp-40h]
  __int64 v46; // [rsp-40h] [rbp-40h]
  __int64 v47; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v9 = *(_QWORD **)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD **)a1;
    if ( *(_QWORD *)a2 == v6 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v45 = *(_DWORD *)(a2 + 8);
      if ( v12 <= v10 )
      {
        v22 = *(_QWORD **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v32 = *(_QWORD **)a1;
          v37 = 3 * v12;
          v33 = a2 + 24 * v12 + 16;
          do
          {
            v34 = *(_QWORD *)(v6 + 16);
            v35 = v32[2];
            if ( v34 != v35 )
            {
              if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
              {
                v43 = v6;
                sub_BD60C0(v32);
                v6 = v43;
              }
              v32[2] = v34;
              if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
              {
                v44 = v6;
                sub_BD73F0((__int64)v32);
                v6 = v44;
              }
            }
            v6 += 24;
            v32 += 3;
          }
          while ( v6 != v33 );
          v22 = *(_QWORD **)a1;
          v10 = *(unsigned int *)(a1 + 8);
          v11 = &v9[v37];
        }
        v23 = &v22[3 * v10];
        while ( v11 != v23 )
        {
          v24 = *(v23 - 1);
          v23 -= 3;
          if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
          {
            v40 = v11;
            sub_BD60C0(v23);
            v11 = v40;
          }
        }
        *(_DWORD *)(a1 + 8) = v45;
        v25 = *(_QWORD **)a2;
        v26 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v26 )
        {
          do
          {
            v27 = *(v26 - 1);
            v26 -= 3;
            if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
              sub_BD60C0(v26);
          }
          while ( v25 != v26 );
        }
      }
      else
      {
        if ( v12 > *(unsigned int *)(a1 + 12) )
        {
          v30 = &v9[3 * v10];
          while ( v30 != v9 )
          {
            while ( 1 )
            {
              v31 = *(v30 - 1);
              v30 -= 3;
              LOBYTE(v6) = v31 != 0;
              if ( v31 == 0 || v31 == -4096 || v31 == -8192 )
                break;
              sub_BD60C0(v30);
              if ( v30 == v9 )
                goto LABEL_55;
            }
          }
LABEL_55:
          *(_DWORD *)(a1 + 8) = 0;
          v10 = 0;
          sub_D6B530(a1, v12, a3, v6, (__int64)v11, a6);
          v6 = *(_QWORD *)a2;
          v12 = *(unsigned int *)(a2 + 8);
          v9 = *(_QWORD **)a1;
          v13 = *(_QWORD **)a2;
        }
        else
        {
          v13 = (_QWORD *)v6;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v36 = 24 * v10;
            v10 *= 24LL;
            do
            {
              v28 = *(_QWORD *)(v6 + 16);
              v29 = v11[2];
              if ( v28 != v29 )
              {
                if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
                {
                  v38 = v6;
                  v41 = v11;
                  sub_BD60C0(v11);
                  v6 = v38;
                  v11 = v41;
                }
                v11[2] = v28;
                if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
                {
                  v39 = v6;
                  v42 = v11;
                  sub_BD73F0((__int64)v11);
                  v6 = v39;
                  v11 = v42;
                }
              }
              v6 += 24;
              v11 += 3;
            }
            while ( v6 != a2 + v10 + 16 );
            v6 = *(_QWORD *)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v9 = *(_QWORD **)a1;
            v13 = (_QWORD *)(*(_QWORD *)a2 + v36);
          }
        }
        v14 = (_QWORD *)((char *)v9 + v10);
        for ( i = (_QWORD *)(v6 + 24 * v12); i != v13; v14 += 3 )
        {
          if ( v14 )
          {
            *v14 = 0;
            v14[1] = 0;
            v16 = v13[2];
            v14[2] = v16;
            if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
              sub_BD6050(v14, *v13 & 0xFFFFFFFFFFFFFFF8LL);
          }
          v13 += 3;
        }
        *(_DWORD *)(a1 + 8) = v45;
        v17 = *(_QWORD **)a2;
        v18 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v18 )
        {
          do
          {
            v19 = *(v18 - 1);
            v18 -= 3;
            if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
              sub_BD60C0(v18);
          }
          while ( v17 != v18 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v20 = &v9[3 * v10];
      if ( v20 != v9 )
      {
        do
        {
          v21 = *(v20 - 1);
          v20 -= 3;
          if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
          {
            v46 = v6;
            sub_BD60C0(v20);
            v6 = v46;
          }
        }
        while ( v20 != v9 );
        v11 = *(_QWORD **)a1;
      }
      if ( v11 != (_QWORD *)(a1 + 16) )
      {
        v47 = v6;
        _libc_free((unsigned __int64)v11);
        v6 = v47;
      }
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
