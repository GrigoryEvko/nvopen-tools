// Function: sub_191D600
// Address: 0x191d600
//
__int64 __fastcall sub_191D600(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdi
  unsigned __int16 v12; // ax
  unsigned __int64 v13; // rdx
  double v14; // xmm4_8
  double v15; // xmm5_8
  __int64 v16; // rcx
  unsigned int v17; // r13d
  __int64 *v19; // r15
  double v20; // xmm4_8
  double v21; // xmm5_8
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  __int64 *v25; // rdi
  char v26; // al
  int v27; // r8d
  int v28; // r9d
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v31; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v32[6]; // [rsp+10h] [rbp-30h] BYREF

  v11 = *(_QWORD *)a1;
  if ( !v11 )
    return 0;
  v12 = *(_WORD *)(a2 + 18);
  if ( ((v12 >> 7) & 6) != 0 || (v12 & 1) != 0 )
    return 0;
  if ( !*(_QWORD *)(a2 + 8) )
  {
    sub_190ACD0(a1 + 152, a2);
    v29 = *(unsigned int *)(a1 + 680);
    if ( (unsigned int)v29 >= *(_DWORD *)(a1 + 684) )
    {
      sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v27, v28);
      v29 = *(unsigned int *)(a1 + 680);
    }
    v17 = 1;
    *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v29) = a2;
    ++*(_DWORD *)(a1 + 680);
    return v17;
  }
  v13 = sub_141C430(v11, a2, 1u);
  if ( (v13 & 7) != 3 )
  {
    if ( (v13 & 7) - 1 <= 1 )
    {
      v16 = *(_QWORD *)(a2 - 24);
      v32[0] = 0;
      v17 = sub_190C3B0(a1, (__int64 *)a2, v13, v16, (__int64)v32, 1u);
      if ( (_BYTE)v17 )
      {
        v19 = (__int64 *)sub_190AE30((__int64)v32, (__int64 ***)a2, a2, (__int64 *)a1);
        sub_1909530(a2, (__int64)v19);
        sub_164D160(a2, (__int64)v19, a3, a4, a5, a6, v20, v21, a9, a10);
        sub_190ACD0(a1 + 152, a2);
        v24 = *(unsigned int *)(a1 + 680);
        if ( (unsigned int)v24 >= *(_DWORD *)(a1 + 684) )
        {
          sub_16CD150(a1 + 672, (const void *)(a1 + 688), 0, 8, v22, v23);
          v24 = *(unsigned int *)(a1 + 680);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v24) = a2;
        v25 = *(__int64 **)(a1 + 104);
        ++*(_DWORD *)(a1 + 680);
        v30 = a2;
        v31 = v19;
        sub_190E970(v25, &v30, (__int64 *)&v31);
        if ( *(_QWORD *)a1 )
        {
          v26 = *(_BYTE *)(*v19 + 8);
          if ( v26 == 16 )
            v26 = *(_BYTE *)(**(_QWORD **)(*v19 + 16) + 8LL);
          if ( v26 == 15 )
            sub_14134C0(*(_QWORD *)a1, v19);
        }
        return v17;
      }
    }
    return 0;
  }
  if ( v13 >> 61 != 1 )
    return 0;
  return sub_191D5A0(a1, (__int64 *)a2, a3, a4, a5, a6, v14, v15, a9, a10);
}
