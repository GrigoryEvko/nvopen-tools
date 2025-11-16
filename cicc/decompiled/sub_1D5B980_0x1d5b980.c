// Function: sub_1D5B980
// Address: 0x1d5b980
//
__int64 __fastcall sub_1D5B980(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r13
  _QWORD *v12; // rax
  double v13; // xmm4_8
  double v14; // xmm5_8
  _QWORD *v15; // r14
  __int64 v16; // r15
  _QWORD *v17; // rbx
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // r10
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 result; // rax
  _QWORD *v24; // rdx
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 i; // [rsp+8h] [rbp-48h]
  const void *v27; // [rsp+10h] [rbp-40h]

  v11 = a2;
  v12 = (_QWORD *)sub_22077B0(96);
  v15 = v12;
  if ( v12 )
  {
    v12[1] = a2;
    v16 = *(_QWORD *)(a2 + 8);
    *v12 = &off_4985678;
    v27 = v12 + 4;
    v12[2] = v12 + 4;
    v12[3] = 0x400000000LL;
    for ( i = (__int64)(v12 + 2); v16; v16 = *(_QWORD *)(v16 + 8) )
    {
      v17 = sub_1648700(v16);
      v20 = (unsigned int)sub_1648720(v16);
      v21 = *((unsigned int *)v15 + 6);
      if ( (unsigned int)v21 >= *((_DWORD *)v15 + 7) )
      {
        v25 = v20;
        sub_16CD150(i, v27, 0, 16, v18, v19);
        v21 = *((unsigned int *)v15 + 6);
        v20 = v25;
      }
      v22 = (_QWORD *)(v15[2] + 16 * v21);
      *v22 = v17;
      v22[1] = v20;
      ++*((_DWORD *)v15 + 6);
    }
    a2 = a3;
    sub_164D160(v11, a3, a4, a5, a6, a7, v13, v14, a10, a11);
  }
  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
  {
    sub_1D5B850(a1, a2);
    result = *(unsigned int *)(a1 + 8);
  }
  v24 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)result);
  if ( v24 )
  {
    *v24 = v15;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    result = (unsigned int)(result + 1);
    *(_DWORD *)(a1 + 8) = result;
    if ( v15 )
      return (*(__int64 (__fastcall **)(_QWORD *))(*v15 + 8LL))(v15);
  }
  return result;
}
