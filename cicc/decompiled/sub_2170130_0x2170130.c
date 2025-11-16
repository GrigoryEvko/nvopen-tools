// Function: sub_2170130
// Address: 0x2170130
//
__int64 __fastcall sub_2170130(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10,
        _QWORD *a11)
{
  __int64 v15; // rsi
  __int64 v16; // r8
  __int64 v17; // rdx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rsi
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r15
  int v27; // edx
  int v28; // r14d
  unsigned __int8 *v29; // rbx
  __int64 v30; // r8
  unsigned int v31; // ecx
  _QWORD *v32; // r13
  int v33; // edx
  int v34; // ebx
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  int v38; // [rsp+0h] [rbp-70h]
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+28h] [rbp-48h] BYREF
  __int64 v43; // [rsp+30h] [rbp-40h] BYREF
  int v44; // [rsp+38h] [rbp-38h]

  if ( *(_DWORD *)(*(_QWORD *)(a11[4] + 16LL) + 252LL) > 0x4Fu )
  {
    v15 = *(_QWORD *)a10;
    v16 = a3;
    v43 = v15;
    if ( v15 )
    {
      v38 = a4;
      v39 = a2;
      sub_1623A60((__int64)&v43, v15, 2);
      a4 = v38;
      a2 = v39;
      v16 = a3;
    }
    v17 = *(_QWORD *)(a2 + 88);
    v44 = *(_DWORD *)(a10 + 8);
    v18 = *(_QWORD **)(v17 + 24);
    if ( *(_DWORD *)(v17 + 32) > 0x40u )
      v18 = (_QWORD *)*v18;
    v19 = *(_QWORD *)(a3 + 88);
    v20 = *(_QWORD *)(v19 + 24);
    if ( *(_DWORD *)(v19 + 32) > 0x40u )
      v20 = **(_QWORD **)(v19 + 24);
    if ( (_DWORD)v18 == 4057 )
    {
      if ( (v20 & 0x1000000000LL) == 0 )
      {
LABEL_10:
        v21 = v43;
        *(_QWORD *)a1 = v16;
        *(_DWORD *)(a1 + 8) = a4;
        *(_QWORD *)(a1 + 16) = a5;
        *(_QWORD *)(a1 + 24) = a6;
        if ( v21 )
          sub_161E7C0((__int64)&v43, v21);
        return a1;
      }
      v36 = sub_1D38BB0((__int64)a11, 0x1000000000LL, (__int64)&v43, 6, 0, 1, a7, a8, a9, 0);
    }
    else
    {
      if ( ((unsigned int)v18 & 0xFFFFFFFB) != 0xFCB )
        goto LABEL_10;
      v42 = sub_1C278B0(0);
      BYTE1(v42) = BYTE1(v42) & 0xFA | v20 & 1 | 4;
      v35 = sub_1C278C0((__int64)&v42);
      v36 = sub_1D38BB0((__int64)a11, v35, (__int64)&v43, 6, 0, 1, a7, a8, a9, 0);
    }
    v16 = v36;
    a4 = v37;
    goto LABEL_10;
  }
  v23 = *(_QWORD *)(a3 + 88);
  v24 = *(_QWORD **)(v23 + 24);
  if ( *(_DWORD *)(v23 + 32) > 0x40u )
    v24 = (_QWORD *)*v24;
  v42 = sub_1C278B0((__int64)v24);
  if ( (v42 & 0x1000000000LL) != 0 )
  {
    v42 = sub_1C278B0(0);
    BYTE4(v42) |= 0x10u;
  }
  BYTE1(v42) &= ~4u;
  v25 = sub_1C278C0((__int64)&v42);
  v26 = sub_1D38BB0((__int64)a11, v25, a10, 6, 0, 1, a7, a8, a9, 0);
  v28 = v27;
  v29 = (unsigned __int8 *)(*(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6);
  v30 = *((_QWORD *)v29 + 1);
  v31 = *v29;
  v43 = 0;
  v44 = 0;
  v32 = sub_1D2B300(a11, 0x30u, (__int64)&v43, v31, v30, (__int64)&v43);
  v34 = v33;
  if ( v43 )
    sub_161E7C0((__int64)&v43, v43);
  *(_QWORD *)a1 = v26;
  *(_DWORD *)(a1 + 8) = v28;
  *(_QWORD *)(a1 + 16) = v32;
  *(_DWORD *)(a1 + 24) = v34;
  return a1;
}
