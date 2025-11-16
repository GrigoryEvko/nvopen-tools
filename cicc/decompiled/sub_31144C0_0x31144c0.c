// Function: sub_31144C0
// Address: 0x31144c0
//
void __fastcall sub_31144C0(unsigned __int64 *a1, __int64 a2, __int64 a3, char a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int64 *v7; // rbx
  int v8; // eax
  _QWORD *v9; // r12
  unsigned __int64 v10; // rcx
  unsigned int v11; // edx
  __int64 v12; // rax
  unsigned __int64 v13; // r13
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 *v16; // r13
  unsigned __int64 v17; // r12
  unsigned __int64 *v18; // r14
  unsigned __int64 v19; // rax
  unsigned __int64 *v20; // r12
  __int64 v21; // r8
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  unsigned __int64 *i; // rax
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rdx
  _QWORD *j; // r13
  unsigned __int64 *v28; // r14
  __int64 v29; // rdx
  unsigned int v30; // eax
  unsigned __int64 **v31; // rdx
  unsigned __int64 *v32; // r12
  __int64 v33; // rbx
  unsigned __int64 v34; // r15
  __int64 v35; // rdx
  unsigned int v36; // eax
  unsigned __int64 *v37; // rdx
  unsigned __int64 v38; // rdx
  __int64 v39; // r8
  unsigned __int64 *v40; // rax
  unsigned __int64 *v43; // [rsp+28h] [rbp-C8h]
  __int64 v44; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v45; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 *v46; // [rsp+38h] [rbp-B8h] BYREF
  _QWORD *v47; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v48; // [rsp+48h] [rbp-A8h]
  unsigned int v49; // [rsp+4Ch] [rbp-A4h]
  _QWORD v50[6]; // [rsp+50h] [rbp-A0h] BYREF
  unsigned __int64 *v51; // [rsp+80h] [rbp-70h] BYREF
  __int64 v52; // [rsp+88h] [rbp-68h]
  _BYTE v53[96]; // [rsp+90h] [rbp-60h] BYREF

  v7 = a1;
  v49 = 6;
  v50[0] = a1;
  v47 = v50;
  v8 = 1;
  while ( 1 )
  {
    v48 = v8 - 1;
    if ( *(_QWORD *)(a2 + 16) )
    {
      v51 = v7;
      (*(void (__fastcall **)(__int64, unsigned __int64 **))(a2 + 24))(a2, &v51);
    }
    if ( a4 )
    {
      v51 = (unsigned __int64 *)v53;
      v52 = 0x300000000LL;
      v9 = (_QWORD *)v7[4];
      if ( v9 )
      {
        v10 = 3;
        v11 = 0;
        while ( 1 )
        {
          v12 = v11;
          v13 = v9[2];
          if ( v11 >= v10 )
          {
            v38 = v11 + 1LL;
            v39 = v9[1];
            if ( v10 < v12 + 1 )
            {
              v44 = v9[1];
              sub_C8D5F0((__int64)&v51, v53, v38, 0x10u, v39, a6);
              v12 = (unsigned int)v52;
              v39 = v44;
            }
            v40 = &v51[2 * v12];
            *v40 = v39;
            v40[1] = v13;
            v11 = v52 + 1;
            LODWORD(v52) = v52 + 1;
          }
          else
          {
            v14 = &v51[2 * v11];
            if ( v14 )
            {
              v15 = v9[1];
              v14[1] = v13;
              *v14 = v15;
              v11 = v52;
            }
            LODWORD(v52) = ++v11;
          }
          v9 = (_QWORD *)*v9;
          if ( !v9 )
            break;
          v10 = HIDWORD(v52);
        }
        v16 = v51;
        v17 = 2LL * v11;
        v18 = &v51[v17];
        if ( v51 != &v51[v17] )
        {
          _BitScanReverse64(&v19, (__int64)(v17 * 8) >> 4);
          sub_3113F40(v51, &v51[v17], 2LL * (int)(63 - (v19 ^ 0x3F)));
          if ( v17 <= 32 )
          {
            sub_31141F0(v16, v18);
          }
          else
          {
            v20 = v16 + 32;
            sub_31141F0(v16, v16 + 32);
            if ( v18 != v16 + 32 )
            {
              do
              {
                v22 = *v20;
                v23 = v20[1];
                for ( i = v20; ; i[3] = v26 )
                {
                  v25 = *(i - 2);
                  if ( v22 >= v25 && (v22 != v25 || v23 >= *(i - 1)) )
                    break;
                  *i = v25;
                  v26 = *(i - 1);
                  i -= 2;
                }
                v20 += 2;
                *i = v22;
                i[1] = v23;
              }
              while ( v18 != v20 );
            }
          }
          v18 = v51;
          v32 = &v51[2 * (unsigned int)v52];
          if ( v32 != v51 )
          {
            v43 = v7;
            v33 = a3;
            do
            {
              v34 = v18[1];
              if ( *(_QWORD *)(v33 + 16) )
              {
                v46 = (unsigned __int64 *)v18[1];
                v45 = v43;
                (*(void (__fastcall **)(__int64, unsigned __int64 **, unsigned __int64 **))(v33 + 24))(v33, &v45, &v46);
              }
              v35 = v48;
              v36 = v48;
              if ( v48 >= (unsigned __int64)v49 )
              {
                a6 = v48 + 1LL;
                if ( v49 < a6 )
                {
                  sub_C8D5F0((__int64)&v47, v50, v48 + 1LL, 8u, v21, a6);
                  v35 = v48;
                }
                v47[v35] = v34;
                ++v48;
              }
              else
              {
                v37 = &v47[v48];
                if ( v37 )
                {
                  *v37 = v34;
                  v36 = v48;
                }
                v48 = v36 + 1;
              }
              v18 += 2;
            }
            while ( v18 != v32 );
            v18 = v51;
            a3 = v33;
          }
        }
        if ( v18 != (unsigned __int64 *)v53 )
          _libc_free((unsigned __int64)v18);
      }
    }
    else
    {
      for ( j = (_QWORD *)v7[4]; j; j = (_QWORD *)*j )
      {
        v28 = (unsigned __int64 *)j[2];
        if ( *(_QWORD *)(a3 + 16) )
        {
          v46 = v7;
          v51 = v28;
          (*(void (__fastcall **)(__int64, unsigned __int64 **, unsigned __int64 **))(a3 + 24))(a3, &v46, &v51);
        }
        v29 = v48;
        v30 = v48;
        if ( v48 >= (unsigned __int64)v49 )
        {
          if ( v49 < (unsigned __int64)v48 + 1 )
          {
            sub_C8D5F0((__int64)&v47, v50, v48 + 1LL, 8u, v48 + 1LL, a6);
            v29 = v48;
          }
          v47[v29] = v28;
          ++v48;
        }
        else
        {
          v31 = (unsigned __int64 **)&v47[v48];
          if ( v31 )
          {
            *v31 = v28;
            v30 = v48;
          }
          v48 = v30 + 1;
        }
      }
    }
    v8 = v48;
    if ( !v48 )
      break;
    v7 = (unsigned __int64 *)v47[v48 - 1];
  }
  if ( v47 != v50 )
    _libc_free((unsigned __int64)v47);
}
