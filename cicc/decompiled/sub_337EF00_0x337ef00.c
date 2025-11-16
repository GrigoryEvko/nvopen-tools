// Function: sub_337EF00
// Address: 0x337ef00
//
void __fastcall sub_337EF00(__int64 a1, __int64 a2, const __m128i **a3)
{
  int v6; // edx
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int16 *v9; // rax
  int v10; // eax
  int v11; // r13d
  __int64 v12; // rdi
  unsigned __int16 *v13; // rax
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rdx
  _QWORD *v17; // rdi
  int v18; // eax
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  const __m128i *v21; // r12
  __int64 v22; // r14
  __int64 v23; // r13
  int v24; // eax
  int v25; // edx
  __int64 v26; // rax
  int v27; // edx
  __int64 v28; // r14
  __int64 v29; // r13
  int v30; // r12d
  _QWORD *v31; // rax
  __int64 v32; // rsi
  __m128i v33; // [rsp+0h] [rbp-D0h]
  __int128 v34; // [rsp+10h] [rbp-C0h]
  __int64 v35; // [rsp+28h] [rbp-A8h]
  char v36; // [rsp+30h] [rbp-A0h]
  __int64 v37; // [rsp+40h] [rbp-90h]
  __int64 v38; // [rsp+48h] [rbp-88h]
  __int64 v39; // [rsp+50h] [rbp-80h] BYREF
  int v40; // [rsp+58h] [rbp-78h]
  __int128 v41; // [rsp+60h] [rbp-70h] BYREF
  __int64 v42; // [rsp+70h] [rbp-60h]
  __int64 v43[10]; // [rsp+80h] [rbp-50h] BYREF

  v6 = *(_DWORD *)(a1 + 848);
  v7 = *(_QWORD *)a1;
  v39 = 0;
  v40 = v6;
  if ( v7 )
  {
    if ( &v39 != (__int64 *)(v7 + 48) )
    {
      v8 = *(_QWORD *)(v7 + 48);
      v39 = v8;
      if ( v8 )
        sub_B96E90((__int64)&v39, v8, 1);
    }
  }
  v35 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v9 = (unsigned __int16 *)(*(_QWORD *)((*a3)->m128i_i64[0] + 48) + 16LL * (*a3)->m128i_u32[2]);
  v38 = *((_QWORD *)v9 + 1);
  v37 = *v9;
  LOWORD(v10) = sub_B5A5E0(a2);
  v11 = v10;
  v36 = BYTE1(v10);
  sub_B91FC0(v43, a2);
  if ( !v36 )
    LOBYTE(v11) = sub_33CC4A0(*(_QWORD *)(a1 + 864), v37, v38);
  v12 = *(_QWORD *)(a1 + 864);
  v33 = _mm_loadu_si128(*a3 + 1);
  v13 = (unsigned __int16 *)(*(_QWORD *)((*a3)[1].m128i_i64[0] + 48) + 16LL * (*a3)[1].m128i_u32[2]);
  v14 = *((_QWORD *)v13 + 1);
  v15 = *v13;
  *(_QWORD *)&v41 = 0;
  DWORD2(v41) = 0;
  *(_QWORD *)&v34 = sub_33F17F0(v12, 51, &v41, v15, v14);
  *((_QWORD *)&v34 + 1) = v16;
  if ( (_QWORD)v41 )
    sub_B91220((__int64)&v41, v41);
  v17 = *(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL);
  *((_QWORD *)&v41 + 1) = 0;
  BYTE4(v42) = 0;
  *(_QWORD *)&v41 = v35 & 0xFFFFFFFFFFFFFFFBLL;
  v18 = 0;
  if ( v35 )
  {
    v19 = *(_QWORD *)(v35 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      v19 = **(_QWORD **)(v19 + 16);
    v18 = *(_DWORD *)(v19 + 8) >> 8;
  }
  LODWORD(v42) = v18;
  v20 = sub_2E7BD70(v17, 2u, -1, v11, (int)v43, 0, v41, v42, 1u, 0, 0);
  v21 = *a3;
  v22 = v20;
  v23 = *(_QWORD *)(a1 + 864);
  v24 = sub_33738A0(a1);
  v26 = sub_33F51B0(
          v23,
          v24,
          v25,
          (unsigned int)&v39,
          v21->m128i_i64[0],
          v21->m128i_i64[1],
          v33.m128i_i64[0],
          v33.m128i_i64[1],
          v34,
          *(_OWORD *)&v21[2],
          *(_OWORD *)&v21[3],
          v37,
          v38,
          v22,
          0,
          0,
          0);
  v28 = *(_QWORD *)(a1 + 864);
  v29 = v26;
  v30 = v27;
  if ( v26 )
  {
    nullsub_1875(v26, *(_QWORD *)(a1 + 864), 0);
    *(_QWORD *)(v28 + 384) = v29;
    *(_DWORD *)(v28 + 392) = v30;
    sub_33E2B60(v28, 0);
  }
  else
  {
    *(_QWORD *)(v28 + 384) = 0;
    *(_DWORD *)(v28 + 392) = v27;
  }
  *(_QWORD *)&v41 = a2;
  v31 = sub_337DC20(a1 + 8, (__int64 *)&v41);
  *v31 = v29;
  v32 = v39;
  *((_DWORD *)v31 + 2) = v30;
  if ( v32 )
    sub_B91220((__int64)&v39, v32);
}
