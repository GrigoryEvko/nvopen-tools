// Function: sub_2FD7440
// Address: 0x2fd7440
//
void __fastcall sub_2FD7440(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r14
  __int32 *v7; // rbx
  __int32 v8; // eax
  _QWORD *v9; // r13
  __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int32 v13; // ecx
  int v14; // edx
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // [rsp+20h] [rbp-B0h]
  __int64 v18; // [rsp+28h] [rbp-A8h]
  __int32 v19; // [rsp+34h] [rbp-9Ch]
  __int64 v20; // [rsp+48h] [rbp-88h] BYREF
  __int64 v21; // [rsp+50h] [rbp-80h] BYREF
  __int64 v22; // [rsp+58h] [rbp-78h]
  __int64 v23; // [rsp+60h] [rbp-70h]
  __m128i v24; // [rsp+70h] [rbp-60h] BYREF
  __int64 v25; // [rsp+80h] [rbp-50h]
  __int64 v26; // [rsp+88h] [rbp-48h]
  __int64 v27; // [rsp+90h] [rbp-40h]

  v6 = (__int64 *)sub_2E313E0(a2);
  v7 = *(__int32 **)a3;
  v17 = *(_QWORD *)(*(_QWORD *)a1 + 8LL) - 800LL;
  v18 = *(_QWORD *)a3 + 12LL * *(unsigned int *)(a3 + 8);
  if ( v18 != *(_QWORD *)a3 )
  {
    do
    {
      v20 = 0;
      v21 = 0;
      v22 = 0;
      v23 = 0;
      v8 = *v7;
      v24.m128i_i64[0] = 0;
      v19 = v8;
      v9 = *(_QWORD **)(a2 + 32);
      v10 = (__int64)sub_2E7B380(v9, v17, (unsigned __int8 **)&v24, 0);
      if ( v24.m128i_i64[0] )
        sub_B91220((__int64)&v24, v24.m128i_i64[0]);
      sub_2E31040((__int64 *)(a2 + 40), v10);
      v11 = *v6;
      v12 = *(_QWORD *)v10;
      *(_QWORD *)(v10 + 8) = v6;
      v11 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = v11 | v12 & 7;
      *(_QWORD *)(v11 + 8) = v10;
      *v6 = v10 | *v6 & 7;
      if ( v22 )
        sub_2E882B0(v10, (__int64)v9, v22);
      if ( v23 )
        sub_2E88680(v10, (__int64)v9, v23);
      v24.m128i_i64[0] = 0x10000000;
      v24.m128i_i32[2] = v19;
      v25 = 0;
      v26 = 0;
      v27 = 0;
      sub_2E8EAD0(v10, (__int64)v9, &v24);
      v13 = v7[1];
      v14 = v7[2] & 0xFFF;
      v25 = 0;
      v24.m128i_i32[2] = v13;
      v24.m128i_i64[0] = (unsigned int)(v14 << 8);
      v26 = 0;
      v27 = 0;
      sub_2E8EAD0(v10, (__int64)v9, &v24);
      if ( v21 )
        sub_B91220((__int64)&v21, v21);
      if ( v20 )
        sub_B91220((__int64)&v20, v20);
      v16 = *(unsigned int *)(a4 + 8);
      if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        sub_C8D5F0(a4, (const void *)(a4 + 16), v16 + 1, 8u, v16 + 1, v15);
        v16 = *(unsigned int *)(a4 + 8);
      }
      v7 += 3;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v16) = v10;
      ++*(_DWORD *)(a4 + 8);
    }
    while ( (__int32 *)v18 != v7 );
  }
}
