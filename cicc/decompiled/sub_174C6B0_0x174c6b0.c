// Function: sub_174C6B0
// Address: 0x174c6b0
//
__int64 __fastcall sub_174C6B0(
        __int64 *a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned int v12; // r14d
  int v13; // ebx
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 **v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 *v27; // r14
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  unsigned __int8 *v34; // [rsp+8h] [rbp-78h] BYREF
  __int64 v35[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v36; // [rsp+20h] [rbp-60h]
  _BYTE v37[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v38; // [rsp+40h] [rbp-40h]

  v10 = *a2;
  v11 = *(_QWORD *)*(a2 - 3);
  if ( *(_BYTE *)(v11 + 8) == 16 )
    v11 = **(_QWORD **)(v11 + 16);
  v12 = *(_DWORD *)(v11 + 8) >> 8;
  v13 = sub_16431D0(*a2);
  if ( v13 == 8 * (unsigned int)sub_15A95A0(a1[333], v12) )
    return sub_174C560(a1, (__int64)a2, a3, a4, a5, a6, v14, v15, a9, a10);
  v16 = a1[333];
  v17 = sub_16498A0((__int64)a2);
  v18 = (__int64 *)sub_15A9620(v16, v17, v12);
  v19 = (__int64 **)v18;
  if ( *(_BYTE *)(v10 + 8) == 16 )
    v19 = (__int64 **)sub_16463B0(v18, *(_DWORD *)(v10 + 32));
  v20 = *(a2 - 3);
  v21 = a1[1];
  v36 = 257;
  if ( v19 != *(__int64 ***)v20 )
  {
    if ( *(_BYTE *)(v20 + 16) > 0x10u )
    {
      v38 = 257;
      v25 = sub_15FDBD0(45, v20, (__int64)v19, (__int64)v37, 0);
      v26 = *(_QWORD *)(v21 + 8);
      v20 = v25;
      if ( v26 )
      {
        v27 = *(__int64 **)(v21 + 16);
        sub_157E9D0(v26 + 40, v25);
        v28 = *(_QWORD *)(v20 + 24);
        v29 = *v27;
        *(_QWORD *)(v20 + 32) = v27;
        v29 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v20 + 24) = v29 | v28 & 7;
        *(_QWORD *)(v29 + 8) = v20 + 24;
        *v27 = *v27 & 7 | (v20 + 24);
      }
      sub_164B780(v20, v35);
      v34 = (unsigned __int8 *)v20;
      if ( !*(_QWORD *)(v21 + 80) )
        sub_4263D6(v20, v35, v30);
      (*(void (__fastcall **)(__int64, unsigned __int8 **))(v21 + 88))(v21 + 64, &v34);
      v31 = *(_QWORD *)v21;
      if ( *(_QWORD *)v21 )
      {
        v34 = *(unsigned __int8 **)v21;
        sub_1623A60((__int64)&v34, v31, 2);
        v32 = *(_QWORD *)(v20 + 48);
        if ( v32 )
          sub_161E7C0(v20 + 48, v32);
        v33 = v34;
        *(_QWORD *)(v20 + 48) = v34;
        if ( v33 )
          sub_1623210((__int64)&v34, v33, v20 + 48);
      }
    }
    else
    {
      v22 = sub_15A46C0(45, (__int64 ***)v20, v19, 0);
      v23 = sub_14DBA30(v22, *(_QWORD *)(v21 + 96), 0);
      if ( v23 )
        v22 = v23;
      v20 = v22;
    }
  }
  v38 = 257;
  return sub_15FE0A0((_QWORD *)v20, v10, 0, (__int64)v37, 0);
}
