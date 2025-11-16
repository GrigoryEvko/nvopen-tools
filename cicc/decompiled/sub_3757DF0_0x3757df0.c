// Function: sub_3757DF0
// Address: 0x3757df0
//
__int64 __fastcall sub_3757DF0(__int64 *a1, unsigned __int8 *a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int32 v8; // r15d
  __int64 v9; // rax
  bool v10; // cc
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int32 v15; // eax
  unsigned __int8 *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 *v19; // rsi
  __int64 v20; // rdi
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 v24; // [rsp+8h] [rbp-A8h]
  __int32 v25; // [rsp+1Ch] [rbp-94h]
  unsigned __int8 *v26; // [rsp+28h] [rbp-88h] BYREF
  unsigned __int8 *v27; // [rsp+30h] [rbp-80h] BYREF
  __int64 v28; // [rsp+38h] [rbp-78h]
  __int64 v29; // [rsp+40h] [rbp-70h] BYREF
  __m128i v30; // [rsp+50h] [rbp-60h] BYREF
  __int64 v31; // [rsp+60h] [rbp-50h]
  __int64 v32; // [rsp+68h] [rbp-48h]
  __int64 v33; // [rsp+70h] [rbp-40h]

  v8 = sub_3752000(a1, **((_QWORD **)a2 + 5), *(_QWORD *)(*((_QWORD *)a2 + 5) + 8LL), (__int64)a3, a5, a6);
  v9 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)a2 + 5) + 40LL) + 96LL);
  v10 = *(_DWORD *)(v9 + 32) <= 0x40u;
  v11 = *(_QWORD **)(v9 + 24);
  if ( !v10 )
    v11 = (_QWORD *)*v11;
  v12 = sub_2FF6410(a1[3], *(_QWORD **)(*(_QWORD *)(a1[3] + 280) + 8LL * (unsigned int)v11));
  v15 = sub_2EC06C0(a1[1], (__int64)v12, byte_3F871B3, 0, v13, v14);
  v16 = (unsigned __int8 *)*((_QWORD *)a2 + 10);
  v25 = v15;
  v17 = a1[2];
  v26 = v16;
  v18 = *(_QWORD *)(v17 + 8) - 800LL;
  if ( v16 )
  {
    v24 = *(_QWORD *)(v17 + 8) - 800LL;
    sub_B96E90((__int64)&v26, (__int64)v16, 1);
    v18 = v24;
    v27 = v26;
    if ( v26 )
    {
      sub_B976B0((__int64)&v26, v26, (__int64)&v27);
      v18 = v24;
      v26 = 0;
    }
  }
  else
  {
    v27 = 0;
  }
  v19 = (__int64 *)a1[6];
  v20 = a1[5];
  v28 = 0;
  v29 = 0;
  v21 = sub_2F26260(v20, v19, (__int64 *)&v27, v18, v25);
  v30.m128i_i32[2] = v8;
  v30.m128i_i64[0] = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  sub_2E8EAD0(v22, (__int64)v21, &v30);
  if ( v27 )
    sub_B91220((__int64)&v27, (__int64)v27);
  if ( v26 )
    sub_B91220((__int64)&v26, (__int64)v26);
  v27 = a2;
  LODWORD(v28) = 0;
  LODWORD(v29) = v25;
  return sub_3755010((__int64)&v30, a3, (unsigned __int64 *)&v27, &v29);
}
