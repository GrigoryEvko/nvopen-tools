// Function: sub_203D1B0
// Address: 0x203d1b0
//
__int64 __fastcall sub_203D1B0(__int64 **a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  char v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rax
  char v16; // di
  __int64 v17; // rax
  unsigned int v18; // ebx
  unsigned int v19; // ecx
  __int64 v20; // r12
  unsigned int v22; // ecx
  bool v23; // al
  unsigned int v24; // eax
  const void **v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // r12
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r13
  __int64 (__fastcall *v31)(__int64, __int64); // r14
  __int64 v32; // rdi
  unsigned int v33; // edx
  unsigned __int8 v34; // al
  __int128 v35; // rax
  __int128 v36; // [rsp-10h] [rbp-80h]
  __int64 v37; // [rsp-8h] [rbp-78h]
  unsigned int v38; // [rsp+8h] [rbp-68h]
  __int64 *v39; // [rsp+8h] [rbp-68h]
  unsigned int v40; // [rsp+10h] [rbp-60h] BYREF
  const void **v41; // [rsp+18h] [rbp-58h]
  char v42[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v43; // [rsp+28h] [rbp-48h]
  __int64 v44; // [rsp+30h] [rbp-40h] BYREF
  int v45; // [rsp+38h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_BYTE *)v7;
  v41 = *(const void ***)(v7 + 8);
  v9 = *(_QWORD *)(a2 + 32);
  LOBYTE(v40) = v8;
  v10 = sub_20363F0((__int64)a1, *(_QWORD *)v9, *(_QWORD *)(v9 + 8));
  v11 = *(_QWORD *)(a2 + 72);
  v13 = v12;
  v14 = v10;
  v15 = *(_QWORD *)(v10 + 40) + 16LL * (unsigned int)v12;
  v16 = *(_BYTE *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v44 = v11;
  v42[0] = v16;
  v43 = v17;
  if ( v11 )
  {
    sub_1623A60((__int64)&v44, v11, 2);
    v16 = v42[0];
  }
  v45 = *(_DWORD *)(a2 + 64);
  if ( v16 )
    v18 = sub_2021900(v16);
  else
    v18 = sub_1F58D40((__int64)v42);
  if ( v8 )
  {
    v19 = sub_2021900(v8);
    if ( v18 % v19 || (unsigned __int8)(v8 - 14) <= 0x60u )
      goto LABEL_8;
  }
  else
  {
    v22 = sub_1F58D40((__int64)&v40);
    if ( v18 % v22 )
      goto LABEL_8;
    v38 = v22;
    v23 = sub_1F58D20((__int64)&v40);
    v19 = v38;
    if ( v23 )
      goto LABEL_8;
  }
  v24 = sub_1F7DEB0((_QWORD *)a1[1][6], v40, (__int64)v41, v18 / v19, 0);
  if ( !(_BYTE)v24 || !(*a1)[(unsigned __int8)v24 + 15] )
  {
LABEL_8:
    v20 = sub_200D7B0((__int64)a1, v14, v13, v40, (__int64)v41);
    goto LABEL_9;
  }
  *((_QWORD *)&v36 + 1) = v13;
  *(_QWORD *)&v36 = v14;
  v26 = sub_1D309E0(a1[1], 158, (__int64)&v44, v24, v25, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v36);
  v27 = a1[1];
  v28 = v26;
  v30 = v29;
  v39 = *a1;
  v31 = *(__int64 (__fastcall **)(__int64, __int64))(**a1 + 48);
  v32 = sub_1E0A0C0(v27[4]);
  if ( v31 == sub_1D13A20 )
  {
    v33 = 8 * sub_15A9520(v32, 0);
    if ( v33 == 32 )
    {
      v34 = 5;
    }
    else if ( v33 > 0x20 )
    {
      v34 = 6;
      if ( v33 != 64 )
      {
        v34 = 0;
        if ( v33 == 128 )
          v34 = 7;
      }
    }
    else
    {
      v34 = 3;
      if ( v33 != 8 )
        v34 = 4 * (v33 == 16);
    }
  }
  else
  {
    v34 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64))v31)(v39, v32, v37);
  }
  *(_QWORD *)&v35 = sub_1D38BB0((__int64)v27, 0, (__int64)&v44, v34, 0, 0, a3, a4, a5, 0);
  v20 = (__int64)sub_1D332F0(v27, 106, (__int64)&v44, v40, v41, 0, *(double *)a3.m128i_i64, a4, a5, v28, v30, v35);
LABEL_9:
  if ( v44 )
    sub_161E7C0((__int64)&v44, v44);
  return v20;
}
