// Function: sub_7F8CF0
// Address: 0x7f8cf0
//
void __fastcall sub_7F8CF0(__int64 a1, int *a2, const __m128i *a3, __int64 *a4, __int64 *a5)
{
  unsigned int v8; // r13d
  __int64 v9; // rsi
  __m128i *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdi
  _QWORD *v14; // r15
  _QWORD *v15; // rax
  void *v16; // rax
  __int64 *v17; // rcx
  __m128i *v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __m128i *v26; // rax
  __m128i *v27; // r14
  _BYTE *v28; // r15
  _QWORD *v29; // rbx
  void *v30; // r14
  __m128i v31; // xmm1
  unsigned __int8 v32; // al
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 v35; // r15
  _QWORD *v36; // rax
  _QWORD *v37; // [rsp+8h] [rbp-98h]
  _QWORD *v38; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+28h] [rbp-78h] BYREF
  int v41[8]; // [rsp+30h] [rbp-70h] BYREF
  _OWORD v42[5]; // [rsp+50h] [rbp-50h] BYREF

  if ( unk_4F06884 || (v32 = sub_622A90(0x40u, 0), v32 == 13) )
    v8 = 5;
  else
    v8 = v32;
  v38 = sub_72BA30(v8);
  if ( (*(_BYTE *)(a1 + 172) & 0x20) != 0 && (*(_BYTE *)(a1 + 88) & 0x70) != 0x10
    || (unsigned int)sub_736A50(*(_QWORD *)(qword_4F04C50 + 32LL))
    || (*(_BYTE *)(a1 + 170) & 0x10) != 0 )
  {
    v9 = v8;
    v10 = sub_7E22A0("_ZGV", v8);
    *a5 = (__int64)v10;
    v10[10].m128i_i8[13] |= 0x80u;
    v11 = *(_QWORD *)(a1 + 240);
    if ( v11 )
      *(_QWORD *)(*a5 + 240) = v11;
    *(_BYTE *)(*a5 + 168) = *(_BYTE *)(a1 + 168) & 7 | *(_BYTE *)(*a5 + 168) & 0xF8;
    v12 = *a5;
  }
  else
  {
    v9 = 0;
    v18 = sub_7E9300((__int64)v38, 0);
    *a5 = (__int64)v18;
    v12 = (__int64)v18;
  }
  *(_BYTE *)(v12 + 176) = *(_BYTE *)(a1 + 176) & 8 | *(_BYTE *)(v12 + 176) & 0xF7;
  if ( (*(_BYTE *)(a1 + 140) & 1) != 0 )
    *(_DWORD *)(*a5 + 140) |= 1u;
  v13 = *a5;
  if ( unk_4F06884 )
  {
    v37 = sub_73E830(v13);
    v37[2] = sub_73A830(1, 5u);
    v14 = sub_73DBF0(0x37u, (__int64)v38, (__int64)v37);
    v14[2] = sub_73A830(0, 5u);
  }
  else
  {
    v19 = sub_73E230(v13, v9);
    v20 = sub_7E23D0(v19);
    v21 = sub_73DCD0(v20);
    v14 = sub_731370((__int64)v21, v9, v22, v23, v24, v25);
    v14[2] = sub_73A830(0, 0);
  }
  v15 = sub_72BA30(5u);
  v16 = sub_73DBF0(0x3Au, (__int64)v15, (__int64)v14);
  v17 = &v40;
  if ( !dword_4F06888 )
    v17 = a4;
  sub_7F8BA0((__int64)v16, 1, a2, v17, (__int64)a3, 0);
  if ( dword_4F06888 )
  {
    v26 = (__m128i *)sub_73E230(*a5, 1);
    v27 = v26;
    if ( qword_4F18AF0 )
    {
      v28 = sub_7F88E0(qword_4F18AF0, v26);
    }
    else
    {
      v35 = sub_72D2E0(*(_QWORD **)(*a5 + 120));
      v36 = sub_72BA30(5u);
      v28 = sub_7F8B20("__cxa_guard_acquire", &qword_4F18AF0, (__int64)v36, v35, 0, v27);
    }
    if ( qword_4F18AE8 )
    {
      v29 = sub_7F88E0(qword_4F18AE8, v27);
    }
    else
    {
      v33 = sub_72D2E0(*(_QWORD **)(*a5 + 120));
      v34 = sub_72CBE0();
      v29 = sub_7F8B20("__cxa_guard_release", &qword_4F18AE8, v34, v33, 0, v27);
    }
    v30 = sub_7F0830(v28);
    sub_7E1740(v40, (__int64)v41);
    sub_7F8BA0((__int64)v30, 1, v41, a4, (__int64)a3, 0);
    v31 = _mm_loadu_si128(a3 + 1);
    v42[0] = _mm_loadu_si128(a3);
    v42[1] = v31;
    sub_7E69E0(v29, (int *)v42);
  }
}
