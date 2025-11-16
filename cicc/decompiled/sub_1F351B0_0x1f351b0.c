// Function: sub_1F351B0
// Address: 0x1f351b0
//
void __fastcall sub_1F351B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r14
  __int32 *v7; // rbx
  __int32 v8; // r11d
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  __int32 v13; // ecx
  int v14; // edx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rdx
  __int64 v18; // [rsp+18h] [rbp-98h]
  __int64 v19; // [rsp+20h] [rbp-90h]
  __int32 v20; // [rsp+34h] [rbp-7Ch]
  __int64 v21; // [rsp+48h] [rbp-68h] BYREF
  __m128i v22; // [rsp+50h] [rbp-60h] BYREF
  __int64 v23; // [rsp+60h] [rbp-50h]
  __int64 v24; // [rsp+68h] [rbp-48h]
  __int64 v25; // [rsp+70h] [rbp-40h]

  v6 = (__int64 *)sub_1DD5EE0(a2);
  v7 = *(__int32 **)a3;
  v18 = *(_QWORD *)(*(_QWORD *)a1 + 8LL) + 960LL;
  v19 = *(_QWORD *)a3 + 12LL * *(unsigned int *)(a3 + 8);
  if ( v19 != *(_QWORD *)a3 )
  {
    do
    {
      v8 = *v7;
      v21 = 0;
      v9 = *(_QWORD *)(a2 + 56);
      v20 = v8;
      v10 = (__int64)sub_1E0B640(v9, v18, &v21, 0);
      sub_1DD5BA0((__int64 *)(a2 + 16), v10);
      v11 = *(_QWORD *)v10;
      v12 = *v6;
      *(_QWORD *)(v10 + 8) = v6;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v10;
      *v6 = v10 | *v6 & 7;
      v22.m128i_i64[0] = 0x10000000;
      v22.m128i_i32[2] = v20;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      sub_1E1A9C0(v10, v9, &v22);
      v13 = v7[1];
      v14 = v7[2] & 0xFFF;
      v23 = 0;
      v22.m128i_i32[2] = v13;
      v22.m128i_i64[0] = (unsigned int)(v14 << 8);
      v24 = 0;
      v25 = 0;
      sub_1E1A9C0(v10, v9, &v22);
      if ( v21 )
        sub_161E7C0((__int64)&v21, v21);
      v17 = *(unsigned int *)(a4 + 8);
      if ( (unsigned int)v17 >= *(_DWORD *)(a4 + 12) )
      {
        sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v15, v16);
        v17 = *(unsigned int *)(a4 + 8);
      }
      v7 += 3;
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v17) = v10;
      ++*(_DWORD *)(a4 + 8);
    }
    while ( (__int32 *)v19 != v7 );
  }
}
