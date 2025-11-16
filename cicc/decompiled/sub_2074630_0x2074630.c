// Function: sub_2074630
// Address: 0x2074630
//
void __fastcall sub_2074630(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 *v5; // r12
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 *v9; // rax
  unsigned int v10; // edx
  unsigned int v11; // ebx
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  int v14; // edx
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // rsi
  __int16 *v18; // rdx
  __int64 v19; // rax
  unsigned __int8 *v20; // rbx
  __int64 v21; // r8
  __int128 v22; // rax
  __int64 *v23; // r14
  int v24; // edx
  int v25; // ebx
  __int64 v26; // r15
  __int64 v27; // [rsp+8h] [rbp-128h]
  unsigned __int64 v28; // [rsp+28h] [rbp-108h]
  __int64 *v29; // [rsp+30h] [rbp-100h]
  __int16 *v30; // [rsp+38h] [rbp-F8h]
  int v32; // [rsp+48h] [rbp-E8h]
  __int64 v33; // [rsp+70h] [rbp-C0h] BYREF
  int v34; // [rsp+78h] [rbp-B8h]
  unsigned __int64 v35[2]; // [rsp+80h] [rbp-B0h] BYREF
  _BYTE v36[32]; // [rsp+90h] [rbp-A0h] BYREF
  unsigned __int64 v37[2]; // [rsp+B0h] [rbp-80h] BYREF
  _BYTE v38[112]; // [rsp+C0h] [rbp-70h] BYREF

  v37[0] = (unsigned __int64)v38;
  v5 = *(__int64 **)(a2 - 48);
  v37[1] = 0x400000000LL;
  v35[1] = 0x400000000LL;
  v6 = *(_QWORD *)(a1 + 552);
  v35[0] = (unsigned __int64)v36;
  v7 = *v5;
  v8 = sub_1E0A0C0(*(_QWORD *)(v6 + 32));
  sub_20C7CE0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL), v8, v7, v37, v35, 0);
  v9 = sub_20685E0(a1, v5, a3, a4, a5);
  v11 = v10;
  v12 = (__int64)v9;
  v13 = sub_1FE5EB0(*(_QWORD *)(a1 + 712), a2);
  v14 = *(_DWORD *)(a1 + 536);
  v33 = 0;
  v32 = v13;
  v15 = *(__int64 **)(a1 + 552);
  v28 = HIDWORD(v13);
  v16 = *(_QWORD *)a1;
  v34 = v14;
  if ( v16 )
  {
    if ( &v33 != (__int64 *)(v16 + 48) )
    {
      v17 = *(_QWORD *)(v16 + 48);
      v33 = v17;
      if ( v17 )
        sub_1623A60((__int64)&v33, v17, 2);
    }
  }
  v29 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v30 = v18;
  v19 = v11;
  v20 = (unsigned __int8 *)(*(_QWORD *)(v12 + 40) + 16LL * v11);
  v27 = v19;
  *(_QWORD *)&v22 = sub_1D2A660(v15, v32, *v20, *((_QWORD *)v20 + 1), v21, 0);
  v23 = sub_1D3A900(
          v15,
          0x2Eu,
          (__int64)&v33,
          1u,
          0,
          0,
          (__m128)a3,
          *(double *)a4.m128i_i64,
          a5,
          (unsigned __int64)v29,
          v30,
          v22,
          v12,
          v27);
  v25 = v24;
  if ( v33 )
    sub_161E7C0((__int64)&v33, v33);
  v26 = *(_QWORD *)(a1 + 552);
  if ( v23 )
  {
    nullsub_686();
    *(_QWORD *)(v26 + 176) = v23;
    *(_DWORD *)(v26 + 184) = v25;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v26 + 176) = 0;
    *(_DWORD *)(v26 + 184) = v25;
  }
  if ( (_BYTE)v28 )
    sub_1FE5190(*(_QWORD *)(a1 + 712), *(_QWORD *)(*(_QWORD *)(a1 + 712) + 784LL), *(_QWORD *)(a2 - 24), v32);
  if ( (_BYTE *)v35[0] != v36 )
    _libc_free(v35[0]);
  if ( (_BYTE *)v37[0] != v38 )
    _libc_free(v37[0]);
}
