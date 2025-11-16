// Function: sub_3826A60
// Address: 0x3826a60
//
void __fastcall sub_3826A60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 *v6; // rax
  __int64 v7; // rsi
  unsigned __int16 v8; // bx
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // rax
  _WORD *v10; // r10
  __int64 *v11; // r15
  __int64 v12; // rax
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  int v15; // eax
  int v16; // esi
  int v17; // eax
  __int64 v18; // r11
  __int64 (__fastcall *v19)(__int64, __int64, unsigned int, __int64); // r9
  __int16 *v20; // rax
  unsigned __int16 v21; // si
  __int64 v22; // r8
  unsigned __int16 v23; // di
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rdx
  unsigned int *v27; // rdi
  __int64 v28; // r9
  char v29; // al
  unsigned __int64 v30; // rdi
  unsigned int *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r9
  unsigned __int8 *v34; // rax
  __int128 v35; // [rsp-10h] [rbp-120h]
  unsigned int v36; // [rsp+10h] [rbp-100h]
  __int64 v37; // [rsp+20h] [rbp-F0h]
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int); // [rsp+28h] [rbp-E8h]
  __int64 v41; // [rsp+40h] [rbp-D0h] BYREF
  int v42; // [rsp+48h] [rbp-C8h]
  unsigned __int8 *v43; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int64 v44; // [rsp+58h] [rbp-B8h]
  _OWORD v45[2]; // [rsp+60h] [rbp-B0h] BYREF
  __int128 v46[2]; // [rsp+80h] [rbp-90h] BYREF
  _QWORD *v47; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-68h]
  _QWORD v49[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v50; // [rsp+C0h] [rbp-50h]

  v6 = *(__int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *v6;
  v9 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v6 + 1);
  v41 = v7;
  v38 = v9;
  if ( v7 )
    sub_B96E90((__int64)&v41, v7, 1);
  v10 = (_WORD *)*a1;
  v11 = (__int64 *)a1[1];
  v42 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = _mm_loadu_si128((const __m128i *)v12);
  v14 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v45[0] = v13;
  v45[1] = v14;
  if ( v8 )
  {
    if ( LOBYTE(v10[250 * v8 + 3240]) == 4 )
    {
      v31 = (unsigned int *)sub_33E5110(v11, v8, (__int64)v38, v8, (__int64)v38);
      *((_QWORD *)&v35 + 1) = 2;
      *(_QWORD *)&v35 = v45;
      v34 = sub_3411630(v11, 66, (__int64)&v41, v31, v32, v33, v35);
      sub_375BC20(a1, (__int64)v34, 1, a3, a4, v13);
      goto LABEL_27;
    }
    v15 = *(_DWORD *)(*(_QWORD *)(v12 + 40) + 24LL);
    if ( v15 != 11 && v15 != 35 )
    {
      v16 = 36;
      if ( v8 == 6 )
        goto LABEL_8;
      goto LABEL_17;
    }
  }
  else
  {
    v17 = *(_DWORD *)(*(_QWORD *)(v12 + 40) + 24LL);
    if ( v17 != 11 )
    {
      v16 = 729;
      if ( v17 != 35 )
        goto LABEL_8;
    }
  }
  v18 = v11[8];
  v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v10 + 592LL);
  v20 = *(__int16 **)(a2 + 48);
  v21 = *v20;
  if ( v19 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v47, (__int64)v10, v18, v21, *((_QWORD *)v20 + 1));
    v22 = v49[0];
    v23 = v48;
    v37 = v49[0];
    v24 = (unsigned __int16)v48;
  }
  else
  {
    v24 = v19((__int64)v10, v18, v21, *((_QWORD *)v20 + 1));
    v37 = v25;
    v23 = v24;
    v22 = v25;
  }
  v36 = v24;
  sub_2FE6CC0((__int64)&v47, *a1, *(_QWORD *)(a1[1] + 64), v23, v22);
  if ( (_BYTE)v47 )
    goto LABEL_16;
  v26 = *(_QWORD *)(a2 + 40);
  LODWORD(v44) = 0;
  *(_QWORD *)&v46[0] = 0;
  DWORD2(v46[0]) = 0;
  v43 = 0;
  sub_375E510((__int64)a1, *(_QWORD *)v26, *(_QWORD *)(v26 + 8), (__int64)&v43, (__int64)v46);
  v27 = (unsigned int *)*a1;
  v28 = a1[1];
  v48 = 0x300000000LL;
  v47 = v49;
  v29 = sub_3474AC0(v27, a2, (__int64)&v47, v36, v37, v28, v13, v43, v44, v46[0]);
  v30 = (unsigned __int64)v47;
  if ( !v29 )
  {
    if ( v47 != v49 )
      _libc_free((unsigned __int64)v47);
LABEL_16:
    v10 = (_WORD *)*a1;
    v11 = (__int64 *)a1[1];
    v16 = 36;
    if ( v8 == 6 )
    {
LABEL_8:
      LOBYTE(v50) = 4;
      v48 = 0;
      v49[0] = 0;
      v49[1] = 0;
      v47 = 0;
      sub_3494590((__int64)v46, v10, (__int64)v11, v16, v8, v38, (__int64)v45, 2u, 0, 0, 0, 0, 4, (__int64)&v41, 0, 0);
      sub_375BC20(a1, *(__int64 *)&v46[0], *((__int64 *)&v46[0] + 1), a3, a4, v13);
      if ( v41 )
        sub_B91220((__int64)&v41, v41);
      return;
    }
LABEL_17:
    v16 = 37;
    if ( v8 != 7 )
    {
      v16 = 38;
      if ( v8 != 8 )
      {
        v16 = 729;
        if ( v8 == 9 )
          v16 = 39;
      }
    }
    goto LABEL_8;
  }
  *(_QWORD *)a3 = *v47;
  *(_DWORD *)(a3 + 8) = *(_DWORD *)(v30 + 8);
  *(_QWORD *)a4 = *(_QWORD *)(v30 + 16);
  *(_DWORD *)(a4 + 8) = *(_DWORD *)(v30 + 24);
  if ( (_QWORD *)v30 != v49 )
    _libc_free(v30);
LABEL_27:
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
}
