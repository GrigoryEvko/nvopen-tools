// Function: sub_1B757D0
// Address: 0x1b757d0
//
__int64 __fastcall sub_1B757D0(
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
  char *v10; // r12
  __int64 v12; // r14
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v20; // rdi
  char *v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rdi
  __int64 v31; // rdx
  char *v32; // [rsp+8h] [rbp-28h] BYREF

  v10 = (char *)a2;
  v12 = *(_QWORD *)a1;
  if ( (**(_BYTE **)a1 & 4) != 0 )
  {
    v15 = sub_1B754F0(*(_QWORD *)a1, a2, a2);
    v16 = *(unsigned int *)(a1 + 16);
    if ( (unsigned int)v16 < *(_DWORD *)(a1 + 20) )
      goto LABEL_3;
LABEL_11:
    sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 8, v13, v14);
    v16 = *(unsigned int *)(a1 + 16);
    goto LABEL_3;
  }
  if ( *(_BYTE *)a2 != 13 )
    goto LABEL_8;
  v20 = (_QWORD *)(*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a2 + 16) & 4) != 0 )
    v20 = (_QWORD *)*v20;
  if ( !(unsigned __int8)sub_16033B0((__int64)v20)
    || (v30 = *(_QWORD *)(a2 + 8 * (7LL - *(unsigned int *)(a2 + 8)))) == 0
    || (sub_161E970(v30), !v31) )
  {
LABEL_8:
    sub_16275A0((__int64 *)&v32, a2);
    v21 = v32;
    v32 = 0;
    v10 = sub_1621440(v21, a2, v22, v23, v24);
    if ( v32 )
      sub_16307F0((__int64)v32, a2, v25, v26, v27, a3, a4, a5, a6, v28, v29, a9, a10);
  }
  v15 = sub_1B754F0(v12, a2, (__int64)v10);
  v16 = *(unsigned int *)(a1 + 16);
  if ( (unsigned int)v16 >= *(_DWORD *)(a1 + 20) )
    goto LABEL_11;
LABEL_3:
  *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * v16) = v15;
  v17 = *(_QWORD *)(a1 + 8);
  v18 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
  *(_DWORD *)(a1 + 16) = v18;
  return *(_QWORD *)(v17 + 8 * v18 - 8);
}
