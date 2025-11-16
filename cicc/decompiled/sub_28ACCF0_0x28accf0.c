// Function: sub_28ACCF0
// Address: 0x28accf0
//
__int64 __fastcall sub_28ACCF0(__int64 a1, __int64 a2, _WORD *a3)
{
  _QWORD *v3; // rbx
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __m128i v6; // xmm2
  __int64 *v7; // rax
  __int64 *v8; // rax
  char v9; // bl
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // ebx
  bool v13; // al
  unsigned __int8 v14; // al
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v19; // rax
  __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int8 v24; // [rsp+Fh] [rbp-351h]
  __m128i v27; // [rsp+20h] [rbp-340h] BYREF
  __m128i v28; // [rsp+30h] [rbp-330h] BYREF
  __m128i v29; // [rsp+40h] [rbp-320h] BYREF
  __m128i v30[3]; // [rsp+50h] [rbp-310h] BYREF
  char v31; // [rsp+80h] [rbp-2E0h]
  _QWORD *v32; // [rsp+90h] [rbp-2D0h] BYREF
  __int64 v33; // [rsp+98h] [rbp-2C8h]
  __int64 v34; // [rsp+A0h] [rbp-2C0h]
  __int64 v35; // [rsp+A8h] [rbp-2B8h] BYREF
  unsigned int v36; // [rsp+B0h] [rbp-2B0h]
  _QWORD v37[2]; // [rsp+1E8h] [rbp-178h] BYREF
  char v38; // [rsp+1F8h] [rbp-168h]
  _BYTE *v39; // [rsp+200h] [rbp-160h]
  __int64 v40; // [rsp+208h] [rbp-158h]
  _BYTE v41[128]; // [rsp+210h] [rbp-150h] BYREF
  __int16 v42; // [rsp+290h] [rbp-D0h]
  _QWORD v43[2]; // [rsp+298h] [rbp-C8h] BYREF
  __int64 v44; // [rsp+2A8h] [rbp-B8h]
  __int64 v45; // [rsp+2B0h] [rbp-B0h] BYREF
  unsigned int v46; // [rsp+2B8h] [rbp-A8h]
  char v47; // [rsp+330h] [rbp-30h] BYREF

  v3 = *(_QWORD **)(a1 + 8);
  sub_D671D0(&v27, a2);
  v4 = _mm_loadu_si128(&v27);
  v5 = _mm_loadu_si128(&v28);
  v31 = 1;
  v6 = _mm_loadu_si128(&v29);
  v32 = v3;
  v7 = &v35;
  v33 = 0;
  v34 = 1;
  v30[0] = v4;
  v30[1] = v5;
  v30[2] = v6;
  do
  {
    *v7 = -4;
    v7 += 5;
    *(v7 - 4) = -3;
    *(v7 - 3) = -4;
    *(v7 - 2) = -3;
  }
  while ( v7 != v37 );
  v40 = 0x400000000LL;
  v37[0] = v43;
  v37[1] = 0;
  v38 = 0;
  v39 = v41;
  v42 = 256;
  v43[1] = 0;
  v44 = 1;
  v43[0] = &unk_49DDBE8;
  v8 = &v45;
  do
  {
    *v8 = -4096;
    v8 += 2;
  }
  while ( v8 != (__int64 *)&v47 );
  v9 = sub_CF63E0(v3, (unsigned __int8 *)a2, v30, (__int64)&v32);
  v43[0] = &unk_49DDBE8;
  if ( (v44 & 1) == 0 )
    sub_C7D6A0(v45, 16LL * v46, 8);
  nullsub_184();
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  if ( (v34 & 1) == 0 )
    sub_C7D6A0(v35, 40LL * v36, 8);
  v10 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (v9 & 2) != 0 )
  {
    v11 = *(_QWORD *)(a2 + 32 * (3 - v10));
    v12 = *(_DWORD *)(v11 + 32);
    if ( v12 <= 0x40 )
      v13 = *(_QWORD *)(v11 + 24) == 0;
    else
      v13 = v12 == (unsigned int)sub_C444A0(v11 + 24);
    if ( v13 && (v14 = sub_28AC6E0(a1, a2)) != 0 )
    {
      v24 = v14;
      *(_QWORD *)a3 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
      a3[4] = 0;
      sub_28AAD10(a1, (_QWORD *)a2, 0, v15, v16, v17);
      return v24;
    }
    else
    {
      return 0;
    }
  }
  else
  {
    v32 = *(_QWORD **)(*(_QWORD *)(a2 - 32 * v10) + 8LL);
    v33 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1 - v10)) + 8LL);
    v34 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2 - v10)) + 8LL);
    v19 = (__int64 *)sub_B43CA0(a2);
    v20 = sub_B6E160(v19, 0xEEu, (__int64)&v32, 3);
    v21 = *(_QWORD *)(a2 - 32) == 0;
    *(_QWORD *)(a2 + 80) = *(_QWORD *)(v20 + 24);
    if ( !v21 )
    {
      v22 = *(_QWORD *)(a2 - 24);
      **(_QWORD **)(a2 - 16) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = *(_QWORD *)(a2 - 16);
    }
    *(_QWORD *)(a2 - 32) = v20;
    v23 = *(_QWORD *)(v20 + 16);
    *(_QWORD *)(a2 - 24) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = a2 - 24;
    *(_QWORD *)(a2 - 16) = v20 + 16;
    *(_QWORD *)(v20 + 16) = a2 - 32;
    return 1;
  }
}
