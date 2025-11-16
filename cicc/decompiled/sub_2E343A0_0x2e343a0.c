// Function: sub_2E343A0
// Address: 0x2e343a0
//
__int64 __fastcall sub_2E343A0(_QWORD *a1, int a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 (*v9)(void); // rax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int32 v12; // eax
  __int64 v13; // rsi
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 *v16; // r8
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rax
  __m128i *v21; // rsi
  unsigned int v22; // ebx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-A8h]
  __int64 *v25; // [rsp+8h] [rbp-A8h]
  unsigned __int32 v26; // [rsp+10h] [rbp-A0h]
  __int64 v27; // [rsp+10h] [rbp-A0h]
  char v28; // [rsp+1Fh] [rbp-91h]
  __int64 v29; // [rsp+28h] [rbp-88h] BYREF
  __int64 v30; // [rsp+30h] [rbp-80h] BYREF
  __int64 v31; // [rsp+38h] [rbp-78h]
  __int64 v32; // [rsp+40h] [rbp-70h]
  __m128i v33; // [rsp+50h] [rbp-60h] BYREF
  __int64 v34; // [rsp+60h] [rbp-50h]
  __int64 v35; // [rsp+68h] [rbp-48h]
  __int64 v36; // [rsp+70h] [rbp-40h]

  v3 = 0;
  v28 = sub_2E31DD0((__int64)a1, a2, -1, -1);
  v6 = sub_2E31210((__int64)a1, a1[7]);
  v7 = a1[4];
  v8 = *(_QWORD *)(v7 + 32);
  v9 = *(__int64 (**)(void))(**(_QWORD **)(v7 + 16) + 128LL);
  if ( v9 != sub_2DAC790 )
  {
    v27 = v6;
    v23 = v9();
    v6 = v27;
    v3 = v23;
  }
  if ( v28 )
  {
    v10 = a1 + 6;
    if ( a1 + 6 != (_QWORD *)v6 )
    {
      while ( *(_WORD *)(v6 + 68) == 20 )
      {
        v11 = *(_QWORD *)(v6 + 32);
        if ( *(_DWORD *)(v11 + 48) == a2 )
        {
          v22 = *(_DWORD *)(v11 + 8);
          if ( !sub_2EBE590(v8, v22, a3, 0) )
            BUG();
          return v22;
        }
        if ( (*(_BYTE *)v6 & 4) != 0 )
        {
          v6 = *(_QWORD *)(v6 + 8);
          if ( v10 == (_QWORD *)v6 )
            break;
        }
        else
        {
          while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
            v6 = *(_QWORD *)(v6 + 8);
          v6 = *(_QWORD *)(v6 + 8);
          if ( v10 == (_QWORD *)v6 )
            break;
        }
      }
    }
  }
  v24 = v6;
  v12 = sub_2EC06C0(v8, a3, byte_3F871B3, 0);
  v13 = *(_QWORD *)(v3 + 8);
  v14 = a1[4];
  v26 = v12;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33.m128i_i64[0] = 0;
  v15 = sub_2E7B380(v14, v13 - 800, &v33, 0);
  v16 = (__int64 *)v24;
  v17 = v15;
  if ( v33.m128i_i64[0] )
  {
    sub_B91220((__int64)&v33, v33.m128i_i64[0]);
    v16 = (__int64 *)v24;
  }
  v25 = v16;
  sub_2E31040(a1 + 5, v17);
  v18 = *v25;
  v19 = *(_QWORD *)v17 & 7LL;
  *(_QWORD *)(v17 + 8) = v25;
  v18 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v17 = v18 | v19;
  *(_QWORD *)(v18 + 8) = v17;
  *v25 = v17 | *v25 & 7;
  if ( v31 )
    sub_2E882B0(v17, v14);
  if ( v32 )
    sub_2E88680(v17, v14);
  v33.m128i_i64[0] = 0x10000000;
  v33.m128i_i32[2] = v26;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  sub_2E8EAD0(v17, v14, &v33);
  v33.m128i_i64[0] = 0x40000000;
  v34 = 0;
  v33.m128i_i32[2] = a2;
  v35 = 0;
  v36 = 0;
  sub_2E8EAD0(v17, v14, &v33);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  if ( !v28 )
  {
    v33.m128i_i64[1] = -1;
    v21 = (__m128i *)a1[24];
    v33.m128i_i32[0] = (unsigned __int16)a2;
    v34 = -1;
    if ( v21 == (__m128i *)a1[25] )
    {
      sub_2E341F0(a1 + 23, v21, &v33);
    }
    else
    {
      if ( v21 )
      {
        *v21 = _mm_loadu_si128(&v33);
        v21[1].m128i_i64[0] = v34;
        v21 = (__m128i *)a1[24];
      }
      a1[24] = (char *)v21 + 24;
    }
  }
  return v26;
}
