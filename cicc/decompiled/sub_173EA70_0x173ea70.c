// Function: sub_173EA70
// Address: 0x173ea70
//
unsigned __int8 *__fastcall sub_173EA70(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 *a9)
{
  __int64 v11; // r14
  __int64 *v12; // rdi
  __int64 *v13; // rdx
  int v14; // esi
  __int64 v15; // rax
  unsigned __int8 *v16; // rax
  unsigned __int8 *v17; // r12
  __int64 v18; // r15
  __int64 v19; // rdi
  unsigned __int64 *v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  bool v24; // zf
  __int64 v25; // rsi
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  unsigned int v32; // [rsp+20h] [rbp-80h]
  unsigned __int8 *v34; // [rsp+48h] [rbp-58h] BYREF
  char v35[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v36; // [rsp+60h] [rbp-40h]

  v36 = 257;
  v11 = *(_QWORD *)(*(_QWORD *)a2 + 24LL);
  v12 = &a7[7 * a8];
  if ( v12 == a7 )
  {
    v14 = 0;
  }
  else
  {
    v13 = a7;
    v14 = 0;
    do
    {
      v15 = v13[5] - v13[4];
      v13 += 7;
      v14 += v15 >> 3;
    }
    while ( v12 != v13 );
  }
  v32 = v14 + a6 + 3;
  v16 = (unsigned __int8 *)sub_1648AB0(72, v32, 16 * (int)a8);
  v17 = v16;
  if ( v16 )
  {
    v18 = (__int64)v16;
    sub_15F1EA0((__int64)v16, **(_QWORD **)(v11 + 16), 5, (__int64)&v16[-24 * v32], v32, 0);
    *((_QWORD *)v17 + 7) = 0;
    sub_15F6500((__int64)v17, v11, a2, a3, a4, (__int64)v35, a5, a6, a7, a8);
  }
  else
  {
    v18 = 0;
  }
  v19 = *(_QWORD *)(a1 + 8);
  if ( v19 )
  {
    v20 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v19 + 40, (__int64)v17);
    v21 = *((_QWORD *)v17 + 3);
    v22 = *v20;
    *((_QWORD *)v17 + 4) = v20;
    v22 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)v17 + 3) = v22 | v21 & 7;
    *(_QWORD *)(v22 + 8) = v17 + 24;
    *v20 = *v20 & 7 | (unsigned __int64)(v17 + 24);
  }
  sub_164B780(v18, a9);
  v24 = *(_QWORD *)(a1 + 80) == 0;
  v34 = v17;
  if ( v24 )
    sub_4263D6(v18, a9, v23);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v34);
  v25 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v34 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v34, v25, 2);
    v26 = *((_QWORD *)v17 + 6);
    if ( v26 )
      sub_161E7C0((__int64)(v17 + 48), v26);
    v27 = v34;
    *((_QWORD *)v17 + 6) = v34;
    if ( v27 )
      sub_1623210((__int64)&v34, v27, (__int64)(v17 + 48));
  }
  return v17;
}
