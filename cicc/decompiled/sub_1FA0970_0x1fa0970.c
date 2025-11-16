// Function: sub_1FA0970
// Address: 0x1fa0970
//
void __fastcall sub_1FA0970(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        __m128i a9,
        unsigned int a10)
{
  unsigned int v10; // r15d
  __int64 v13; // rsi
  _QWORD **v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rbx
  const __m128i *v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // rdi
  __int64 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // r12
  bool v28; // zf
  _QWORD *v29; // rax
  __int64 v30; // r10
  __int64 v31; // r11
  __int64 v32; // rax
  __int64 *v33; // rax
  __int128 v34; // [rsp-10h] [rbp-120h]
  __int128 v35; // [rsp-10h] [rbp-120h]
  __int64 v36; // [rsp+0h] [rbp-110h]
  __int64 v37; // [rsp+8h] [rbp-108h]
  __int64 v38; // [rsp+18h] [rbp-F8h]
  _QWORD **i; // [rsp+28h] [rbp-E8h]
  __int64 v41; // [rsp+40h] [rbp-D0h]
  __int64 v43; // [rsp+50h] [rbp-C0h]
  __int64 v44; // [rsp+70h] [rbp-A0h] BYREF
  int v45; // [rsp+78h] [rbp-98h]
  __int64 v46[2]; // [rsp+80h] [rbp-90h] BYREF
  _BYTE *v47; // [rsp+90h] [rbp-80h] BYREF
  __int64 v48; // [rsp+98h] [rbp-78h]
  _BYTE v49[112]; // [rsp+A0h] [rbp-70h] BYREF

  v13 = *(_QWORD *)(a5 + 72);
  v43 = a5;
  v38 = a6;
  v44 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v44, v13, 2);
  v14 = *(_QWORD ***)a2;
  v45 = *(_DWORD *)(v43 + 64);
  for ( i = &v14[*(unsigned int *)(a2 + 8)]; i != v14; ++v14 )
  {
    v27 = *v14;
    v19 = 0;
    v47 = v49;
    v28 = v27[6] == 0;
    v48 = 0x400000000LL;
    if ( !v28 )
    {
      do
      {
        v29 = (_QWORD *)(v19 + v27[4]);
        v30 = *v29;
        v31 = v29[1];
        if ( *v29 == a3 && *((_DWORD *)v29 + 2) == a4 )
        {
          v32 = (unsigned int)v48;
          if ( (unsigned int)v48 >= HIDWORD(v48) )
          {
            sub_16CD150((__int64)&v47, v49, 0, 16, a5, a6);
            v32 = (unsigned int)v48;
          }
          v33 = (__int64 *)&v47[16 * v32];
          *v33 = v43;
          v33[1] = v38;
          LODWORD(v48) = v48 + 1;
        }
        else
        {
          v15 = *(_QWORD *)(v43 + 40);
          LOBYTE(v10) = *(_BYTE *)v15;
          *((_QWORD *)&v34 + 1) = v31;
          *(_QWORD *)&v34 = v30;
          a5 = sub_1D309E0(
                 *a1,
                 a10,
                 (__int64)&v44,
                 v10,
                 *(const void ***)(v15 + 8),
                 0,
                 *(double *)a7.m128_u64,
                 a8,
                 *(double *)a9.m128i_i64,
                 v34);
          a6 = v16;
          v17 = (unsigned int)v48;
          if ( (unsigned int)v48 >= HIDWORD(v48) )
          {
            v37 = v16;
            v36 = a5;
            sub_16CD150((__int64)&v47, v49, 0, 16, a5, v16);
            v17 = (unsigned int)v48;
            a5 = v36;
            a6 = v37;
          }
          v18 = (__int64 *)&v47[16 * v17];
          *v18 = a5;
          v18[1] = a6;
          LODWORD(v48) = v48 + 1;
        }
        v19 += 40;
      }
      while ( v19 != 80 );
      v20 = (const __m128i *)v27[4];
      v21 = (unsigned int)v48;
      if ( (unsigned int)v48 >= HIDWORD(v48) )
      {
        sub_16CD150((__int64)&v47, v49, 0, 16, a5, a6);
        v21 = (unsigned int)v48;
      }
      a7 = (__m128)_mm_loadu_si128(v20 + 5);
      v22 = v41;
      *(__m128 *)&v47[16 * v21] = a7;
      v23 = *a1;
      LODWORD(v48) = v48 + 1;
      v24 = v27[5];
      LOBYTE(v22) = *(_BYTE *)v24;
      *((_QWORD *)&v35 + 1) = (unsigned int)v48;
      *(_QWORD *)&v35 = v47;
      v41 = v22;
      v25 = sub_1D359D0(
              v23,
              137,
              (__int64)&v44,
              (unsigned int)v22,
              *(const void ***)(v24 + 8),
              0,
              *(double *)a7.m128_u64,
              a8,
              a9,
              v35);
      v46[1] = v26;
      v46[0] = (__int64)v25;
      sub_1F994A0((__int64)a1, (__int64)v27, v46, 1, 1);
      if ( v47 != v49 )
        _libc_free((unsigned __int64)v47);
    }
  }
  if ( v44 )
    sub_161E7C0((__int64)&v44, v44);
}
