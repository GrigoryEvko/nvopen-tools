// Function: sub_24E5090
// Address: 0x24e5090
//
__int64 __fastcall sub_24E5090(__int64 a1, __int64 a2, const __m128i *a3, __int64 *a4, __int64 a5)
{
  int v6; // eax
  unsigned __int64 v7; // r14
  __m128i v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 *v16; // rax
  __int64 v17; // r12
  _QWORD *v18; // rax
  __int64 *v19; // rax
  __int64 v21; // [rsp+8h] [rbp-98h]
  __m128i v22; // [rsp+10h] [rbp-90h] BYREF
  __int16 v23; // [rsp+30h] [rbp-70h]
  __m128i v24[6]; // [rsp+40h] [rbp-60h] BYREF

  v21 = *(_QWORD *)(a1 + 40);
  v6 = *(_DWORD *)(a2 + 280);
  if ( v6 == 3 )
  {
    v17 = *(_QWORD *)(a5 + 8);
    v18 = (_QWORD *)sub_B2BE50(*(_QWORD *)(*(_QWORD *)(a5 + 40) + 72LL));
    v19 = (__int64 *)sub_BCB120(v18);
    v7 = sub_BCF480(v19, *(const void **)(v17 + 16), *(unsigned int *)(v17 + 12), 0);
  }
  else if ( v6 )
  {
    if ( (unsigned int)(v6 - 1) > 1 )
      BUG();
    v7 = *(_QWORD *)(*(_QWORD *)(a2 + 328) + 24LL);
  }
  else
  {
    v24[0].m128i_i64[0] = sub_BCE3C0(**(__int64 ***)(a2 + 288), 0);
    v16 = (__int64 *)sub_BCB120(**(_QWORD ***)(a2 + 288));
    v7 = sub_BCF480(v16, v24, 1, 0);
  }
  v8.m128i_i64[0] = (__int64)sub_BD5D20(a1);
  v23 = 261;
  v22 = v8;
  sub_9C6370(v24, &v22, a3, 261, v9, v10);
  v11 = sub_BD2DA0(136);
  v12 = v11;
  if ( v11 )
    sub_B2C3B0(v11, v7, 7, 0xFFFFFFFF, (__int64)v24, 0);
  sub_BA8540(v21 + 24, v12);
  v13 = *a4;
  v14 = *(_QWORD *)(v12 + 56);
  *(_QWORD *)(v12 + 64) = a4;
  v13 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v12 + 56) = v13 | v14 & 7;
  *(_QWORD *)(v13 + 8) = v12 + 56;
  *a4 = *a4 & 7 | (v12 + 56);
  return v12;
}
