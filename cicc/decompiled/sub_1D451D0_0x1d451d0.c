// Function: sub_1D451D0
// Address: 0x1d451d0
//
void __fastcall sub_1D451D0(__int64 a1, __int64 *a2, __int64 *a3, int a4)
{
  __int64 v7; // rcx
  __int64 *v8; // r8
  __int64 v9; // r15
  __int64 v10; // r9
  __m128i *v11; // rdx
  int v12; // eax
  __int64 i; // r14
  __int64 v14; // rdx
  int v15; // esi
  _BYTE *v16; // rdi
  __int64 v17; // r14
  char *v18; // rbx
  unsigned __int64 v19; // rax
  __m128i *v20; // r14
  __int64 v21; // rcx
  __int32 v22; // edi
  __m128i *v23; // rax
  __int64 v24; // rsi
  __m128i v25; // xmm0
  unsigned int v26; // ecx
  __int64 v27; // r14
  __int64 *v28; // rbx
  __int64 v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 *v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 *v37; // [rsp+0h] [rbp-E0h]
  int v38; // [rsp+8h] [rbp-D8h]
  int v39; // [rsp+Ch] [rbp-D4h]
  char *src; // [rsp+18h] [rbp-C8h]
  unsigned int srca; // [rsp+18h] [rbp-C8h]
  unsigned int srcb; // [rsp+18h] [rbp-C8h]
  __m128i v43; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+30h] [rbp-B0h]
  _BYTE *v45; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v46; // [rsp+48h] [rbp-98h]
  _BYTE v47[144]; // [rsp+50h] [rbp-90h] BYREF

  if ( a4 == 1 )
  {
    sub_1D44C70(a1, *a2, a2[1], *a3, a3[1]);
  }
  else
  {
    sub_1D306C0(a1, *a2, a2[1], *a3, a3[1], 0, 0, 1);
    v45 = v47;
    v46 = 0x400000000LL;
    if ( a4 )
    {
      v8 = a2;
      v9 = 0;
      LODWORD(v10) = 0;
      do
      {
        v11 = (__m128i *)*v8;
        v12 = *((_DWORD *)v8 + 2);
        for ( i = *(_QWORD *)(*v8 + 48); i; LODWORD(v46) = v15 + 1 )
        {
          while ( v12 != *(_DWORD *)(i + 8) )
          {
            i = *(_QWORD *)(i + 32);
            if ( !i )
              goto LABEL_11;
          }
          v14 = *(_QWORD *)(i + 16);
          v43.m128i_i32[2] = v10;
          v44 = i;
          v43.m128i_i64[0] = v14;
          if ( HIDWORD(v46) <= (unsigned int)v9 )
          {
            v37 = v8;
            v38 = v12;
            v39 = v10;
            sub_16CD150((__int64)&v45, v47, 0, 24, (int)v8, v10);
            v9 = (unsigned int)v46;
            v8 = v37;
            v12 = v38;
            LODWORD(v10) = v39;
          }
          v11 = (__m128i *)&v45[24 * v9];
          v7 = v44;
          *v11 = _mm_loadu_si128(&v43);
          v15 = v46;
          v11[1].m128i_i64[0] = v7;
          i = *(_QWORD *)(i + 32);
          v9 = (unsigned int)(v15 + 1);
        }
LABEL_11:
        v10 = (unsigned int)(v10 + 1);
        v8 += 2;
      }
      while ( a4 != (_DWORD)v10 );
      v16 = v45;
      v17 = 24LL * (unsigned int)v9;
      v18 = &v45[v17];
      if ( &v45[v17] != v45 )
      {
        src = v45;
        _BitScanReverse64(&v19, 0xAAAAAAAAAAAAAAABLL * (v17 >> 3));
        sub_1D13550((__int64)v45, (__m128i *)&v45[v17], 2LL * (int)(63 - (v19 ^ 0x3F)), v7, (__int64)v8, v10);
        if ( (unsigned __int64)v17 <= 0x180 )
        {
          sub_1D138A0(src, v18);
          LODWORD(v9) = v46;
          v16 = v45;
        }
        else
        {
          v20 = (__m128i *)(src + 384);
          sub_1D138A0(src, src + 384);
          if ( v18 != src + 384 )
          {
            do
            {
              v21 = v20->m128i_i64[0];
              v22 = v20->m128i_i32[2];
              v23 = (__m128i *)((char *)v20 - 24);
              v24 = v20[1].m128i_i64[0];
              if ( v20[-2].m128i_i64[1] <= v20->m128i_i64[0] )
              {
                v11 = v20;
              }
              else
              {
                do
                {
                  v25 = _mm_loadu_si128(v23);
                  v23[2].m128i_i64[1] = v23[1].m128i_i64[0];
                  v11 = v23;
                  v23 = (__m128i *)((char *)v23 - 24);
                  v23[3] = v25;
                }
                while ( v23->m128i_i64[0] > v21 );
              }
              v20 = (__m128i *)((char *)v20 + 24);
              v11->m128i_i64[0] = v21;
              v11->m128i_i32[2] = v22;
              v11[1].m128i_i64[0] = v24;
            }
            while ( v18 != (char *)v20 );
          }
          LODWORD(v9) = v46;
          v16 = v45;
        }
      }
      if ( (_DWORD)v9 )
      {
        v26 = 0;
        v27 = 0;
LABEL_21:
        srca = v26;
        v28 = *(__int64 **)&v16[24 * v27];
        sub_1D2D480(a1, (__int64)v28, (unsigned int)v11);
        v29 = (__int64)v45;
        v30 = srca;
        while ( 1 )
        {
          ++v30;
          v31 = v29 + 24 * v27;
          v32 = *(_QWORD *)(v31 + 16);
          v33 = &a3[2 * *(unsigned int *)(v31 + 8)];
          if ( *(_QWORD *)v32 )
          {
            v34 = *(_QWORD *)(v32 + 32);
            **(_QWORD **)(v32 + 24) = v34;
            if ( v34 )
              *(_QWORD *)(v34 + 24) = *(_QWORD *)(v32 + 24);
          }
          *(_QWORD *)v32 = *v33;
          *(_DWORD *)(v32 + 8) = *((_DWORD *)v33 + 2);
          v35 = *v33;
          if ( v35 )
          {
            v36 = *(_QWORD *)(v35 + 48);
            *(_QWORD *)(v32 + 32) = v36;
            if ( v36 )
              *(_QWORD *)(v36 + 24) = v32 + 32;
            *(_QWORD *)(v32 + 24) = v35 + 48;
            *(_QWORD *)(v35 + 48) = v32;
          }
          if ( v30 == (_DWORD)v9 )
            break;
          v27 = v30;
          v29 = (__int64)v45;
          if ( *(__int64 **)&v45[24 * v30] != v28 )
          {
            srcb = v30;
            sub_1D446C0(a1, v28);
            v16 = v45;
            v26 = srcb;
            goto LABEL_21;
          }
        }
        sub_1D446C0(a1, v28);
        v16 = v45;
      }
      if ( v16 != v47 )
        _libc_free((unsigned __int64)v16);
    }
  }
}
