// Function: sub_2789F90
// Address: 0x2789f90
//
unsigned __int64 __fastcall sub_2789F90(const __m128i *a1, __int64 a2, unsigned __int64 a3, _QWORD *a4)
{
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 *v6; // rax
  __int64 v7; // r13
  int v8; // ebx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  unsigned __int64 v15; // r15
  __int64 v17; // [rsp+0h] [rbp-350h]
  __int64 v19; // [rsp+28h] [rbp-328h]
  __m128i v20[3]; // [rsp+30h] [rbp-320h] BYREF
  char v21; // [rsp+60h] [rbp-2F0h]
  _QWORD *v22; // [rsp+70h] [rbp-2E0h]
  _QWORD v23[2]; // [rsp+78h] [rbp-2D8h] BYREF
  __int64 v24; // [rsp+88h] [rbp-2C8h]
  __int64 v25; // [rsp+90h] [rbp-2C0h] BYREF
  unsigned int v26; // [rsp+98h] [rbp-2B8h]
  _QWORD v27[2]; // [rsp+1D0h] [rbp-180h] BYREF
  char v28; // [rsp+1E0h] [rbp-170h]
  _BYTE *v29; // [rsp+1E8h] [rbp-168h]
  __int64 v30; // [rsp+1F0h] [rbp-160h]
  _BYTE v31[128]; // [rsp+1F8h] [rbp-158h] BYREF
  __int16 v32; // [rsp+278h] [rbp-D8h]
  _QWORD v33[2]; // [rsp+280h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+290h] [rbp-C0h]
  __int64 v35; // [rsp+298h] [rbp-B8h] BYREF
  unsigned int v36; // [rsp+2A0h] [rbp-B0h]
  char v37; // [rsp+318h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a3 + 40);
  v22 = a4;
  v23[0] = a4;
  v23[1] = 0;
  v24 = 1;
  v19 = v4;
  v5 = &v25;
  do
  {
    *v5 = -4;
    v5 += 5;
    *(v5 - 4) = -3;
    *(v5 - 3) = -4;
    *(v5 - 2) = -3;
  }
  while ( v5 != v27 );
  v27[1] = 0;
  v27[0] = v33;
  v29 = v31;
  v30 = 0x400000000LL;
  v32 = 256;
  v28 = 0;
  v33[1] = 0;
  v34 = 1;
  v33[0] = &unk_49DDBE8;
  v6 = &v35;
  do
  {
    *v6 = -4096;
    v6 += 2;
  }
  while ( v6 != (__int64 *)&v37 );
  if ( v19 )
  {
    v7 = v19;
    v8 = 0;
    while ( 1 )
    {
      if ( v19 == v7 )
      {
        v17 = v7;
        v10 = a3;
        v11 = a2;
        goto LABEL_13;
      }
      v9 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v9 != v7 + 48 )
      {
        if ( !v9 )
          BUG();
        v10 = v9 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 <= 0xA )
          break;
      }
LABEL_20:
      v7 = sub_AA54C0(v7);
      if ( !v7 )
        goto LABEL_21;
    }
    v17 = v7;
    v11 = a2;
LABEL_13:
    while ( ++v8 <= (unsigned int)qword_4FFB968 )
    {
      v12 = _mm_loadu_si128(a1);
      v13 = _mm_loadu_si128(a1 + 1);
      v14 = _mm_loadu_si128(a1 + 2);
      v21 = 1;
      v20[0] = v12;
      v20[1] = v13;
      v20[2] = v14;
      if ( (sub_CF63E0(v22, (unsigned __int8 *)v10, v20, (__int64)v23) & 2) != 0 )
        break;
      if ( *(_BYTE *)v10 == 61 && a1->m128i_i64[0] == *(_QWORD *)(v10 - 32) && v11 == *(_QWORD *)(v10 + 8) )
      {
        v15 = v10;
        goto LABEL_22;
      }
      v10 = sub_B46BC0(v10, 0);
      if ( !v10 )
      {
        v7 = v17;
        goto LABEL_20;
      }
    }
  }
LABEL_21:
  v15 = 0;
LABEL_22:
  v33[0] = &unk_49DDBE8;
  if ( (v34 & 1) == 0 )
    sub_C7D6A0(v35, 16LL * v36, 8);
  nullsub_184();
  if ( v29 != v31 )
    _libc_free((unsigned __int64)v29);
  if ( (v24 & 1) == 0 )
    sub_C7D6A0(v25, 40LL * v26, 8);
  return v15;
}
