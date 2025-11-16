// Function: sub_3417D60
// Address: 0x3417d60
//
void __fastcall sub_3417D60(__int64 a1, __int64 *a2, __int64 *a3, int a4)
{
  __int64 v4; // r15
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 (__fastcall ***v9)(); // rdi
  __int64 v10; // rdx
  __int64 *v11; // r14
  __int64 (__fastcall ***v12)(); // r9
  int v13; // r13d
  __int64 v14; // rbx
  int v15; // r12d
  __int64 (__fastcall **v16)(); // rax
  unsigned __int64 v17; // r8
  const __m128i *v18; // rax
  __m128i *v19; // rdx
  int v20; // eax
  unsigned __int64 v21; // rdx
  int v22; // r14d
  __int64 v23; // rax
  unsigned int v24; // r12d
  __int64 v25; // rbx
  __int64 v26; // rsi
  char *v27; // rax
  char *v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 (__fastcall ***v34)(); // [rsp+0h] [rbp-F0h]
  __int64 (__fastcall ***v35)(); // [rsp+0h] [rbp-F0h]
  __int64 (__fastcall ***v36)(); // [rsp+8h] [rbp-E8h]
  signed __int64 v37; // [rsp+8h] [rbp-E8h]
  __int64 (__fastcall ***v38)(); // [rsp+8h] [rbp-E8h]
  __int64 *v39; // [rsp+10h] [rbp-E0h]
  __int64 (__fastcall ***v40)(); // [rsp+18h] [rbp-D8h]
  __int64 (__fastcall **v42)(); // [rsp+30h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+38h] [rbp-B8h]
  __int64 v44; // [rsp+40h] [rbp-B0h]
  void **p_base; // [rsp+48h] [rbp-A8h]
  void *base; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+58h] [rbp-98h]
  _BYTE v48[144]; // [rsp+60h] [rbp-90h] BYREF

  v4 = a1;
  if ( a4 == 1 )
  {
    sub_34161C0(a1, *a2, a2[1], *a3, a3[1]);
  }
  else
  {
    sub_33F9B80(a1, *a2, a2[1], *a3, a3[1], 0, 0, 1);
    sub_34151B0(a1, *a2, *a3, v6, v7, v8);
    v9 = (__int64 (__fastcall ***)())v48;
    v47 = 0x400000000LL;
    base = v48;
    if ( a4 )
    {
      v10 = 0;
      v11 = a2;
      v39 = a3;
      v12 = &v42;
      v13 = 0;
      do
      {
        v14 = *(_QWORD *)(*v11 + 56);
        if ( v14 )
        {
          v15 = *((_DWORD *)v11 + 2);
          do
          {
            while ( *(_DWORD *)(v14 + 8) != v15 )
            {
              v14 = *(_QWORD *)(v14 + 32);
              if ( !v14 )
                goto LABEL_10;
            }
            v16 = *(__int64 (__fastcall ***)())(v14 + 16);
            v17 = v10 + 1;
            LODWORD(v43) = v13;
            v44 = v14;
            v42 = v16;
            v18 = (const __m128i *)v12;
            if ( v10 + 1 > (unsigned __int64)HIDWORD(v47) )
            {
              if ( v9 > v12 )
              {
                v35 = v12;
                v38 = v12;
                sub_C8D5F0((__int64)&base, v48, v10 + 1, 0x18u, v17, (__int64)v12);
                v9 = (__int64 (__fastcall ***)())base;
                v10 = (unsigned int)v47;
                v18 = (const __m128i *)v38;
                v12 = v35;
              }
              else
              {
                v36 = v12;
                v34 = v12;
                if ( v12 >= &v9[3 * v10] )
                {
                  sub_C8D5F0((__int64)&base, v48, v17, 0x18u, v17, (__int64)v12);
                  v9 = (__int64 (__fastcall ***)())base;
                  v10 = (unsigned int)v47;
                  v12 = v34;
                  v18 = (const __m128i *)v36;
                }
                else
                {
                  v37 = (char *)v12 - (char *)v9;
                  sub_C8D5F0((__int64)&base, v48, v17, 0x18u, v17, (__int64)v12);
                  v9 = (__int64 (__fastcall ***)())base;
                  v10 = (unsigned int)v47;
                  v12 = v34;
                  v18 = (const __m128i *)((char *)base + v37);
                }
              }
            }
            v19 = (__m128i *)&v9[3 * v10];
            *v19 = _mm_loadu_si128(v18);
            v9 = (__int64 (__fastcall ***)())base;
            v19[1].m128i_i64[0] = v18[1].m128i_i64[0];
            v14 = *(_QWORD *)(v14 + 32);
            v10 = (unsigned int)(v47 + 1);
            LODWORD(v47) = v47 + 1;
          }
          while ( v14 );
        }
LABEL_10:
        ++v13;
        v11 += 2;
      }
      while ( a4 != v13 );
      v20 = v10;
      v21 = 3LL * (unsigned int)v10;
      v22 = v20;
      if ( v21 > 3 )
      {
        v40 = v12;
        qsort(v9, 0xAAAAAAAAAAAAAAABLL * ((__int64)(8 * v21) >> 3), 0x18u, (__compar_fn_t)sub_33C7E50);
        v22 = v47;
        v9 = (__int64 (__fastcall ***)())base;
        v12 = v40;
      }
      v23 = *(_QWORD *)(v4 + 768);
      v44 = v4;
      v42 = off_4A36780;
      v43 = v23;
      *(_QWORD *)(v4 + 768) = v12;
      p_base = &base;
      if ( v22 )
      {
        v24 = 0;
        do
        {
          while ( 1 )
          {
            v25 = v24;
            v26 = (__int64)v9[3 * v24];
            if ( v26 )
              break;
            if ( ++v24 == v22 )
              goto LABEL_28;
          }
          sub_33EB970(v4, v26, v21);
          v27 = (char *)base;
          do
          {
            ++v24;
            v28 = &v27[24 * v25];
            v29 = *((_QWORD *)v28 + 2);
            v30 = &v39[2 * *((unsigned int *)v28 + 2)];
            if ( *(_QWORD *)v29 )
            {
              v31 = *(_QWORD *)(v29 + 32);
              **(_QWORD **)(v29 + 24) = v31;
              if ( v31 )
                *(_QWORD *)(v31 + 24) = *(_QWORD *)(v29 + 24);
            }
            *(_QWORD *)v29 = *v30;
            *(_DWORD *)(v29 + 8) = *((_DWORD *)v30 + 2);
            v32 = *v30;
            if ( v32 )
            {
              v33 = *(_QWORD *)(v32 + 56);
              *(_QWORD *)(v29 + 32) = v33;
              if ( v33 )
                *(_QWORD *)(v33 + 24) = v29 + 32;
              *(_QWORD *)(v29 + 24) = v32 + 56;
              *(_QWORD *)(v32 + 56) = v29;
            }
            if ( v24 == v22 )
              break;
            v25 = v24;
            v27 = (char *)base;
          }
          while ( *((_QWORD *)base + 3 * v24) == v26 );
          sub_3415B20(v4, v26);
          v9 = (__int64 (__fastcall ***)())base;
        }
        while ( v24 != v22 );
LABEL_28:
        v4 = v44;
        v23 = v43;
      }
      *(_QWORD *)(v4 + 768) = v23;
      if ( v9 != (__int64 (__fastcall ***)())v48 )
        _libc_free((unsigned __int64)v9);
    }
  }
}
