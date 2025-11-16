// Function: sub_2F232C0
// Address: 0x2f232c0
//
void __fastcall sub_2F232C0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  _QWORD *v18; // rbx
  __int64 v19; // rsi
  unsigned int v20; // r12d
  char *v21; // r13
  __int64 v22; // rdx
  unsigned int v23; // edi
  __int64 v24; // rdx
  unsigned int v25; // ecx
  char *v26; // rbx
  __m128i *v27; // rdi
  const __m128i *v28; // r13
  __int16 v29; // dx
  __int64 v30; // rdi
  __int64 v31; // rsi
  size_t v32; // rdx
  const __m128i *v33; // r12
  unsigned __int64 *v34; // r14
  unsigned __int64 *v35; // [rsp+0h] [rbp-F0h]
  _QWORD *v36; // [rsp+18h] [rbp-D8h]
  _QWORD *v37; // [rsp+20h] [rbp-D0h]
  char *v39; // [rsp+38h] [rbp-B8h]
  char *v40; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v41; // [rsp+50h] [rbp-A0h]
  char v42[8]; // [rsp+58h] [rbp-98h] BYREF
  unsigned __int64 v43; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int64 *v44; // [rsp+68h] [rbp-88h] BYREF
  __m128i *v45; // [rsp+70h] [rbp-80h]
  const __m128i *v46; // [rsp+78h] [rbp-78h]
  __m128i v47; // [rsp+80h] [rbp-70h] BYREF
  _BYTE v48[16]; // [rsp+90h] [rbp-60h] BYREF
  __m128i v49; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v50; // [rsp+B0h] [rbp-40h]

  v4 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 16) + 200LL))(*(_QWORD *)(a3 + 16));
  if ( *(_DWORD *)(a3 + 704) )
  {
    v14 = v4;
    v15 = *(_QWORD **)(a3 + 696);
    v16 = 4LL * *(unsigned int *)(a3 + 712);
    v17 = &v15[v16];
    v37 = &v15[v16];
    if ( v15 != &v15[v16] )
    {
      while ( 1 )
      {
        v18 = v15;
        if ( *v15 != -8192 && *v15 != -4096 )
          break;
        v15 += 4;
        if ( v17 == v15 )
          goto LABEL_2;
      }
      if ( v15 != v37 )
      {
        v35 = a2 + 60;
        while ( 1 )
        {
          v19 = *v18;
          v41 = 0x100000000LL;
          v20 = *((_DWORD *)v18 + 4);
          v21 = v42;
          v40 = v42;
          v39 = v42;
          if ( v20 && &v40 != v18 + 1 )
          {
            v32 = 8;
            if ( v20 == 1
              || (sub_C8D5F0((__int64)&v40, v42, v20, 8u, v5, v6),
                  v21 = v40,
                  (v32 = 8LL * *((unsigned int *)v18 + 4)) != 0) )
            {
              memcpy(v21, (const void *)v18[1], v32);
              v21 = v40;
            }
            LODWORD(v41) = v20;
            v39 = &v21[8 * v20];
          }
          v46 = 0;
          v22 = *(_QWORD *)(v19 + 24);
          v44 = 0;
          v23 = *(_DWORD *)(v22 + 24);
          v24 = *(_QWORD *)(v22 + 56);
          v45 = 0;
          if ( v19 == v24 )
          {
            v25 = 0;
          }
          else
          {
            v25 = 0;
            do
            {
              v24 = *(_QWORD *)(v24 + 8);
              ++v25;
            }
            while ( v19 != v24 );
          }
          v43 = __PAIR64__(v25, v23);
          if ( v39 != v21 )
          {
            v36 = v18;
            v26 = v21;
            do
            {
              v29 = *((_WORD *)v26 + 2);
              v30 = *(_QWORD *)v26;
              v47.m128i_i64[0] = (__int64)v48;
              v47.m128i_i64[1] = 0;
              v50 = v29;
              v48[0] = 0;
              v49 = 0u;
              sub_2F07630(v30, (__int64)&v47, v14);
              v28 = v45;
              if ( v45 == v46 )
              {
                sub_2F14070((unsigned __int64 *)&v44, v45, &v47);
              }
              else
              {
                if ( v45 )
                {
                  v27 = v45;
                  v45->m128i_i64[0] = (__int64)v45[1].m128i_i64;
                  sub_2F07250(v27->m128i_i64, v47.m128i_i64[0], v47.m128i_i64[0] + v47.m128i_i64[1]);
                  v28[2] = _mm_loadu_si128(&v49);
                  v28[3].m128i_i16[0] = v50;
                  v28 = v45;
                }
                v5 = (__int64)&v28[3].m128i_i64[1];
                v45 = (__m128i *)&v28[3].m128i_u64[1];
              }
              if ( (_BYTE *)v47.m128i_i64[0] != v48 )
                j_j___libc_free_0(v47.m128i_u64[0]);
              v26 += 8;
            }
            while ( v39 != v26 );
            v18 = v36;
          }
          v31 = a2[61];
          if ( v31 == a2[62] )
          {
            sub_2F18600(v35, (char *)v31, &v43);
            v33 = v45;
            v34 = v44;
          }
          else
          {
            if ( v31 )
            {
              *(_QWORD *)v31 = v43;
              *(_QWORD *)(v31 + 8) = v44;
              *(_QWORD *)(v31 + 16) = v45;
              *(_QWORD *)(v31 + 24) = v46;
              a2[61] += 32LL;
              goto LABEL_33;
            }
            v33 = v45;
            v34 = v44;
            a2[61] = 32;
          }
          if ( v33 != (const __m128i *)v34 )
          {
            do
            {
              if ( (unsigned __int64 *)*v34 != v34 + 2 )
                j_j___libc_free_0(*v34);
              v34 += 7;
            }
            while ( v33 != (const __m128i *)v34 );
            v34 = v44;
          }
          if ( v34 )
            j_j___libc_free_0((unsigned __int64)v34);
LABEL_33:
          if ( v40 != v42 )
            _libc_free((unsigned __int64)v40);
          v18 += 4;
          if ( v18 != v37 )
          {
            while ( *v18 == -4096 || *v18 == -8192 )
            {
              v18 += 4;
              if ( v37 == v18 )
                goto LABEL_2;
            }
            if ( v37 != v18 )
              continue;
          }
          break;
        }
      }
    }
  }
LABEL_2:
  v7 = a2[61];
  v8 = a2[60];
  if ( v8 != v7 )
  {
    _BitScanReverse64(&v9, (__int64)(v7 - v8) >> 5);
    sub_2F22A40(a2[60], (__m128i *)a2[61], 2LL * (int)(63 - (v9 ^ 0x3F)));
    if ( (__int64)(v7 - v8) <= 512 )
    {
      sub_2F0BD50(v8, v7, v10);
    }
    else
    {
      v11 = v8 + 512;
      sub_2F0BD50(v8, v8 + 512, v10);
      if ( v7 != v8 + 512 )
      {
        do
        {
          v13 = v11;
          v11 += 32LL;
          sub_2F0B9C0(v13, v8 + 512, v12);
        }
        while ( v7 != v11 );
      }
    }
  }
}
