// Function: sub_1D5C680
// Address: 0x1d5c680
//
__int64 __fastcall sub_1D5C680(
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
  _QWORD *v11; // r14
  _QWORD *v12; // rax
  int v13; // r8d
  int v14; // r9d
  _QWORD *v15; // r15
  __int64 v16; // rax
  bool v17; // zf
  int v18; // ebx
  unsigned int v19; // ebx
  __int64 v20; // rbx
  __int64 v21; // r14
  __int64 v22; // r12
  __int64 ***v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rsi
  unsigned __int64 v29; // rcx
  __int64 v30; // rcx
  _QWORD *v31; // rax
  double v32; // xmm4_8
  double v33; // xmm5_8
  _QWORD *v34; // r13
  __int64 v35; // r12
  _QWORD *v36; // rbx
  int v37; // r8d
  __int64 v38; // r9
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 result; // rax
  _QWORD *v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // [rsp+0h] [rbp-60h]
  __int64 ***v45; // [rsp+8h] [rbp-58h]
  __int64 i; // [rsp+8h] [rbp-58h]
  const void *v47; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+20h] [rbp-40h]

  v11 = (_QWORD *)a2;
  v49 = *(_QWORD *)(a1 + 144);
  v12 = (_QWORD *)sub_22077B0(112);
  v15 = v12;
  if ( v12 )
  {
    v12[1] = a2;
    *v12 = off_49856A8;
    v16 = *(_QWORD *)(a2 + 40);
    v17 = a2 + 24 == *(_QWORD *)(v16 + 48);
    *((_BYTE *)v15 + 24) = a2 + 24 != *(_QWORD *)(v16 + 48);
    if ( v17 )
    {
      v15[2] = v16;
    }
    else
    {
      v43 = 0;
      if ( (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        v43 = (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) - 24;
      v15[2] = v43;
    }
    v18 = *(_DWORD *)(a2 + 20);
    v15[5] = a2;
    v15[4] = &off_4985588;
    v19 = v18 & 0xFFFFFFF;
    v15[6] = v15 + 8;
    v15[7] = 0x400000000LL;
    if ( v19 > 4uLL )
      sub_16CD150((__int64)(v15 + 6), v15 + 8, v19, 8, v13, v14);
    if ( v19 )
    {
      v20 = 24LL * v19;
      v21 = 0;
      v22 = a2;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
        {
          v23 = *(__int64 ****)(*(_QWORD *)(v22 - 8) + v21);
          v24 = *((unsigned int *)v15 + 14);
          if ( (unsigned int)v24 >= *((_DWORD *)v15 + 15) )
            goto LABEL_21;
        }
        else
        {
          v23 = *(__int64 ****)(v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF) + v21);
          v24 = *((unsigned int *)v15 + 14);
          if ( (unsigned int)v24 >= *((_DWORD *)v15 + 15) )
          {
LABEL_21:
            v45 = v23;
            sub_16CD150((__int64)(v15 + 6), v15 + 8, 0, 8, v13, (int)v23);
            v24 = *((unsigned int *)v15 + 14);
            v23 = v45;
          }
        }
        *(_QWORD *)(v15[6] + 8 * v24) = v23;
        ++*((_DWORD *)v15 + 14);
        v25 = sub_1599EF0(*v23);
        if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
          v26 = *(_QWORD *)(v22 - 8);
        else
          v26 = v22 - 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
        v27 = (__int64 *)(v21 + v26);
        if ( *v27 )
        {
          v28 = v27[1];
          v29 = v27[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v29 = v28;
          if ( v28 )
            *(_QWORD *)(v28 + 16) = *(_QWORD *)(v28 + 16) & 3LL | v29;
        }
        *v27 = v25;
        if ( v25 )
        {
          v30 = *(_QWORD *)(v25 + 8);
          v27[1] = v30;
          if ( v30 )
            *(_QWORD *)(v30 + 16) = (unsigned __int64)(v27 + 1) | *(_QWORD *)(v30 + 16) & 3LL;
          v27[2] = (v25 + 8) | v27[2] & 3;
          *(_QWORD *)(v25 + 8) = v27;
        }
        v21 += 24;
        if ( v21 == v20 )
        {
          v11 = (_QWORD *)v22;
          break;
        }
      }
    }
    v15[12] = 0;
    v15[13] = v49;
    if ( a3 )
    {
      v31 = (_QWORD *)sub_22077B0(96);
      v34 = v31;
      if ( v31 )
      {
        v31[1] = v11;
        v35 = v11[1];
        *v31 = &off_4985678;
        v47 = v31 + 4;
        v31[2] = v31 + 4;
        v31[3] = 0x400000000LL;
        for ( i = (__int64)(v31 + 2); v35; v35 = *(_QWORD *)(v35 + 8) )
        {
          v36 = sub_1648700(v35);
          v38 = (unsigned int)sub_1648720(v35);
          v39 = *((unsigned int *)v34 + 6);
          if ( (unsigned int)v39 >= *((_DWORD *)v34 + 7) )
          {
            v44 = v38;
            sub_16CD150(i, v47, 0, 16, v37, v38);
            v39 = *((unsigned int *)v34 + 6);
            v38 = v44;
          }
          v40 = (_QWORD *)(v34[2] + 16 * v39);
          *v40 = v36;
          v40[1] = v38;
          ++*((_DWORD *)v34 + 6);
        }
        sub_164D160((__int64)v11, a3, a4, a5, a6, a7, v32, v33, a10, a11);
      }
      v15[12] = v34;
    }
    a2 = (__int64)v11;
    sub_1412190(v49, (__int64)v11);
    sub_15F2070(v11);
  }
  result = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 12) )
  {
    sub_1D5B850(a1, a2);
    result = *(unsigned int *)(a1 + 8);
  }
  v42 = (_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)result);
  if ( v42 )
  {
    *v42 = v15;
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
