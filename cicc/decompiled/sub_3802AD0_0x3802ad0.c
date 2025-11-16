// Function: sub_3802AD0
// Address: 0x3802ad0
//
__int64 __fastcall sub_3802AD0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r11
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r15
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r8
  unsigned __int16 v12; // di
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int16 v19; // cx
  __int64 v20; // rax
  bool v21; // al
  __int64 v22; // rsi
  _QWORD *v23; // r13
  const __m128i *v24; // rcx
  __int64 v25; // r10
  unsigned __int64 v26; // r11
  __m128i *v27; // r14
  __int64 v29; // [rsp+0h] [rbp-A0h]
  __int64 v30; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v31; // [rsp+8h] [rbp-98h]
  const __m128i *v32; // [rsp+18h] [rbp-88h]
  unsigned __int64 v33; // [rsp+20h] [rbp-80h]
  unsigned __int64 v34; // [rsp+28h] [rbp-78h]
  __int64 v35; // [rsp+30h] [rbp-70h] BYREF
  int v36; // [rsp+38h] [rbp-68h]
  unsigned __int64 v37; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v38; // [rsp+48h] [rbp-58h]
  __int64 v39; // [rsp+50h] [rbp-50h] BYREF
  __int64 v40; // [rsp+58h] [rbp-48h]

  if ( *(_DWORD *)(a2 + 24) == 299 && (*(_BYTE *)(a2 + 33) & 4) == 0 && (*(_WORD *)(a2 + 32) & 0x380) == 0 )
    return sub_3848BD0();
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *a1;
  v6 = *(_QWORD *)(v4 + 80);
  v7 = *(_QWORD *)(v4 + 88);
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v34 = *(_QWORD *)v4;
  v9 = *(_QWORD *)(v4 + 8);
  v10 = *(_QWORD *)(*(_QWORD *)(v4 + 40) + 48LL) + 16LL * *(unsigned int *)(v4 + 48);
  v33 = v9;
  v11 = *(_QWORD *)(v10 + 8);
  v12 = *(_WORD *)v10;
  v13 = a1[1];
  if ( v8 == sub_2D56A50 )
    sub_2FE6CC0((__int64)&v39, v5, *(_QWORD *)(v13 + 64), v12, v11);
  else
    v8(v5, *(_QWORD *)(v13 + 64), v12, v11);
  v14 = *(_QWORD *)(a2 + 40);
  v36 = 0;
  LODWORD(v38) = 0;
  v15 = *(_QWORD *)(v14 + 40);
  v16 = *(_QWORD *)(v14 + 48);
  v35 = 0;
  v17 = *(unsigned int *)(v14 + 48);
  v37 = 0;
  v18 = *(_QWORD *)(v15 + 48) + 16 * v17;
  v19 = *(_WORD *)v18;
  v20 = *(_QWORD *)(v18 + 8);
  LOWORD(v39) = v19;
  v40 = v20;
  if ( v19 )
  {
    if ( (unsigned __int16)(v19 - 2) > 7u
      && (unsigned __int16)(v19 - 17) > 0x6Cu
      && (unsigned __int16)(v19 - 176) > 0x1Fu )
    {
      goto LABEL_7;
    }
LABEL_16:
    sub_375E510((__int64)a1, v15, v16, (__int64)&v35, (__int64)&v37);
    goto LABEL_8;
  }
  v29 = v16;
  v21 = sub_3007070((__int64)&v39);
  v16 = v29;
  if ( v21 )
    goto LABEL_16;
LABEL_7:
  sub_375E6F0((__int64)a1, v15, v16, (__int64)&v35, (__int64)&v37);
LABEL_8:
  v22 = *(_QWORD *)(a2 + 80);
  v23 = (_QWORD *)a1[1];
  v24 = *(const __m128i **)(a2 + 112);
  v25 = *(unsigned __int16 *)(a2 + 96);
  v39 = v22;
  v26 = *(_QWORD *)(a2 + 104);
  if ( v22 )
  {
    v30 = v25;
    v31 = *(_QWORD *)(a2 + 104);
    v32 = v24;
    sub_B96E90((__int64)&v39, v22, 1);
    v25 = v30;
    v26 = v31;
    v24 = v32;
  }
  LODWORD(v40) = *(_DWORD *)(a2 + 72);
  v27 = sub_33F49B0(v23, v34, v33, (__int64)&v39, v37, v38, v6, v7, v25, v26, v24);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return (__int64)v27;
}
