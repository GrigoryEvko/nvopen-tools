// Function: sub_393C750
// Address: 0x393c750
//
void __fastcall sub_393C750(_QWORD *a1)
{
  char *v1; // r12
  char *v2; // r13
  unsigned __int64 v3; // rax
  _QWORD *v4; // r14
  __int64 v5; // r12
  unsigned int v6; // ebx
  unsigned __int64 v7; // r15
  unsigned __int32 v8; // eax
  __int64 v9; // rsi
  unsigned __int64 v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rax
  __m128i *v13; // rsi
  char *v14; // rbx
  unsigned int v15; // ecx
  unsigned int v16; // edx
  char *v17; // rax
  char *v18; // rsi
  char *v19; // rsi
  unsigned __int32 *v20; // [rsp+10h] [rbp-C0h]
  __int64 v21; // [rsp+20h] [rbp-B0h]
  unsigned __int32 v22; // [rsp+3Ch] [rbp-94h]
  unsigned __int32 *v24; // [rsp+48h] [rbp-88h]
  unsigned __int64 v25; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int32 v26; // [rsp+58h] [rbp-78h]
  unsigned __int64 v27; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+68h] [rbp-68h]
  unsigned __int64 v29; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+78h] [rbp-58h]
  __m128i v31; // [rsp+80h] [rbp-50h] BYREF
  __int64 v32; // [rsp+90h] [rbp-40h]

  v1 = (char *)a1[7];
  v2 = (char *)a1[6];
  if ( v2 != v1 )
  {
    _BitScanReverse64(&v3, (v1 - v2) >> 2);
    sub_393C3E0(v2, v1, 2LL * (int)(63 - (v3 ^ 0x3F)));
    if ( v1 - v2 > 64 )
    {
      v14 = v2 + 64;
      sub_393C330((unsigned int *)v2, (unsigned int *)v2 + 16);
      if ( v1 != v2 + 64 )
      {
        v15 = *(_DWORD *)v14;
        v16 = *((_DWORD *)v2 + 15);
        v17 = v2 + 60;
        if ( v16 <= *(_DWORD *)v14 )
          goto LABEL_37;
        while ( 1 )
        {
          do
          {
            *((_DWORD *)v17 + 1) = v16;
            v18 = v17;
            v16 = *((_DWORD *)v17 - 1);
            v17 -= 4;
          }
          while ( v15 < v16 );
          v14 += 4;
          *(_DWORD *)v18 = v15;
          if ( v1 == v14 )
            break;
          while ( 1 )
          {
            v15 = *(_DWORD *)v14;
            v16 = *((_DWORD *)v14 - 1);
            v17 = v14 - 4;
            if ( v16 > *(_DWORD *)v14 )
              break;
LABEL_37:
            v19 = v14;
            v14 += 4;
            *(_DWORD *)v19 = v15;
            if ( v1 == v14 )
              goto LABEL_4;
          }
        }
      }
    }
    else
    {
      sub_393C330((unsigned int *)v2, (unsigned int *)v1);
    }
LABEL_4:
    v4 = a1 + 1;
    v20 = (unsigned __int32 *)a1[7];
    if ( (unsigned __int32 *)a1[6] != v20 )
    {
      v24 = (unsigned __int32 *)a1[6];
      v5 = 0;
      v6 = 0;
      v7 = 0;
      v21 = a1[3];
      do
      {
        v8 = *v24;
        v9 = a1[12];
        v26 = 128;
        v22 = v8;
        sub_16A4EF0((__int64)&v25, v9, 0);
        v28 = 128;
        sub_16A4EF0((__int64)&v27, v22, 0);
        v30 = 128;
        sub_16A4EF0((__int64)&v29, 1000000, 0);
        sub_16A7C10((__int64)&v25, (__int64 *)&v27);
        sub_16A9F90((__int64)&v31, (__int64)&v25, (__int64)&v29);
        if ( v26 > 0x40 && v25 )
          j_j___libc_free_0_0(v25);
        v10 = v31.m128i_i64[0];
        v25 = v31.m128i_i64[0];
        v26 = v31.m128i_u32[2];
        if ( v31.m128i_i32[2] > 0x40u )
          v10 = *(_QWORD *)v31.m128i_i64[0];
        v11 = v21;
        if ( v4 != (_QWORD *)v21 && v7 < v10 )
        {
          do
          {
            v5 = *(_QWORD *)(v11 + 32);
            v6 += *(_DWORD *)(v11 + 40);
            v7 += v5 * *(unsigned int *)(v11 + 40);
            v12 = sub_220EEE0(v11);
            v11 = v12;
          }
          while ( v7 < v10 && v4 != (_QWORD *)v12 );
          v21 = v12;
        }
        v31.m128i_i64[1] = v5;
        v31.m128i_i32[0] = v22;
        v13 = (__m128i *)a1[10];
        v32 = v6;
        if ( v13 == (__m128i *)a1[11] )
        {
          sub_393C5A0(a1 + 9, v13, &v31);
        }
        else
        {
          if ( v13 )
          {
            *v13 = _mm_loadu_si128(&v31);
            v13[1].m128i_i64[0] = v32;
            v13 = (__m128i *)a1[10];
          }
          a1[10] = (char *)v13 + 24;
        }
        if ( v30 > 0x40 && v29 )
          j_j___libc_free_0_0(v29);
        if ( v28 > 0x40 && v27 )
          j_j___libc_free_0_0(v27);
        if ( v26 > 0x40 )
        {
          if ( v25 )
            j_j___libc_free_0_0(v25);
        }
        ++v24;
      }
      while ( v20 != v24 );
    }
  }
}
