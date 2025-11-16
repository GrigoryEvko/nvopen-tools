// Function: sub_2B2D6F0
// Address: 0x2b2d6f0
//
__int64 __fastcall sub_2B2D6F0(__int64 **a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __m128i v6; // rax
  unsigned int v7; // r14d
  __m128i v8; // rax
  unsigned int v9; // eax
  __int32 v10; // edx
  unsigned int v11; // r12d
  __int64 *v12; // rax
  __int64 *v13; // r14
  __int64 v14; // rax
  char v15; // [rsp-89h] [rbp-89h]
  __m128i v16; // [rsp-88h] [rbp-88h] BYREF
  __int64 v17; // [rsp-78h] [rbp-78h]
  __int64 v18; // [rsp-70h] [rbp-70h]
  __int64 v19; // [rsp-68h] [rbp-68h]
  __int64 v20; // [rsp-60h] [rbp-60h]
  __int64 v21; // [rsp-58h] [rbp-58h]
  __int64 v22; // [rsp-50h] [rbp-50h]
  __int16 v23; // [rsp-48h] [rbp-48h]

  if ( *(_BYTE *)a2 <= 0x15u )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(v3 + 24);
  if ( **a1 == v4 )
    return 0;
  v5 = *(_QWORD *)((*a1)[23] + 3344);
  v17 = 0;
  v16 = (__m128i)v5;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 257;
  v15 = sub_9AC470(a2, &v16, 0);
  v6.m128i_i64[0] = sub_9208B0(*(_QWORD *)((*a1)[23] + 3344), **a1);
  v16 = v6;
  v7 = sub_CA1930(&v16);
  v8.m128i_i64[0] = sub_9208B0(*(_QWORD *)((*a1)[23] + 3344), v4);
  v16 = v8;
  v9 = sub_CA1930(&v16);
  v10 = *(_DWORD *)(v3 + 32);
  v11 = (v15 == 0) + 39;
  if ( v7 <= v9 )
    v11 = 38;
  v12 = *a1;
  v13 = (__int64 *)(*a1)[14];
  v16.m128i_i8[4] = *(_BYTE *)(v3 + 8) == 18;
  v16.m128i_i32[0] = v10;
  v14 = sub_BCE1B0((__int64 *)*v12, v16.m128i_i64[0]);
  return sub_DFD060(v13, v11, v14, v3);
}
