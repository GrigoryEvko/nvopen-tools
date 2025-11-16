// Function: sub_24E5240
// Address: 0x24e5240
//
void __fastcall sub_24E5240(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // esi
  __int64 v9; // rax
  __int64 *v10; // rcx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 **v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r10
  __m128i v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rax
  unsigned __int64 v22; // r14
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-118h]
  _QWORD *v31; // [rsp+10h] [rbp-110h]
  __int64 v32; // [rsp+18h] [rbp-108h]
  __int64 v33; // [rsp+28h] [rbp-F8h]
  __m128i v34[2]; // [rsp+30h] [rbp-F0h] BYREF
  char v35; // [rsp+50h] [rbp-D0h]
  char v36; // [rsp+51h] [rbp-CFh]
  __m128i v37; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v38; // [rsp+80h] [rbp-A0h]
  __m128i v39[3]; // [rsp+90h] [rbp-90h] BYREF
  __int64 *v40; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v41; // [rsp+C8h] [rbp-58h]
  _BYTE v42[80]; // [rsp+D0h] [rbp-50h] BYREF

  v8 = 0;
  v41 = 0x400000000LL;
  v9 = 8 * a4;
  v10 = (__int64 *)v42;
  v40 = (__int64 *)v42;
  v11 = v9 >> 3;
  if ( (unsigned __int64)v9 > 0x20 )
  {
    v32 = v9;
    sub_C8D5F0((__int64)&v40, v42, v9 >> 3, 8u, a5, a6);
    v8 = v41;
    v9 = v32;
    v10 = &v40[(unsigned int)v41];
  }
  if ( v9 > 0 )
  {
    v12 = 0;
    do
    {
      v10[v12] = a3[v12];
      ++v12;
    }
    while ( v11 - v12 > 0 );
    v8 = v41;
  }
  v13 = *a3;
  LODWORD(v41) = v8 + v11;
  v14 = *(_QWORD *)(v13 + 40);
  v15 = (__int64 **)sub_BCD420(*(__int64 **)(v13 + 8), (unsigned int)(v8 + v11));
  v16 = sub_AD1300(v15, v40, (unsigned int)v41);
  v17 = *(_QWORD **)(v16 + 8);
  v30 = v16;
  v36 = 1;
  v31 = v17;
  v34[0].m128i_i64[0] = (__int64)".resumers";
  v35 = 3;
  v18.m128i_i64[0] = (__int64)sub_BD5D20(a1);
  v38 = 261;
  v37 = v18;
  sub_9C6370(v39, &v37, v34, (__int64)v39, v19, v20);
  BYTE4(v33) = 0;
  v21 = sub_BD2C40(88, unk_3F0FAE8);
  v22 = (unsigned __int64)v21;
  if ( v21 )
    sub_B30000((__int64)v21, v14, v31, 1, 8, v30, (__int64)v39, 0, 0, v33, 0);
  v23 = (__int64 *)sub_B2BE50(a1);
  v24 = sub_BCE3C0(v23, 0);
  v25 = sub_ADAFB0(v22, v24);
  v26 = *(_QWORD *)(*(_QWORD *)a2 - 32LL * (*(_DWORD *)(*(_QWORD *)a2 + 4LL) & 0x7FFFFFF));
  v27 = v26 + 32 * (3LL - (*(_DWORD *)(v26 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v27 )
  {
    v28 = *(_QWORD *)(v27 + 8);
    **(_QWORD **)(v27 + 16) = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(v27 + 16);
  }
  *(_QWORD *)v27 = v25;
  if ( v25 )
  {
    v29 = *(_QWORD *)(v25 + 16);
    *(_QWORD *)(v27 + 8) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = v27 + 8;
    *(_QWORD *)(v27 + 16) = v25 + 16;
    *(_QWORD *)(v25 + 16) = v27;
  }
  if ( v40 != (__int64 *)v42 )
    _libc_free((unsigned __int64)v40);
}
