// Function: sub_3775050
// Address: 0x3775050
//
unsigned __int8 **__fastcall sub_3775050(
        unsigned __int8 ***a1,
        const void *a2,
        __int64 a3,
        unsigned int a4,
        __m128i a5)
{
  __int64 v6; // r14
  __int64 v8; // r15
  __int64 *v10; // rsi
  unsigned __int8 **v11; // rcx
  __int64 v13; // r8
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r10
  unsigned __int8 *v17; // rax
  int v18; // edx
  int v19; // edi
  unsigned __int8 *v20; // rdx
  unsigned __int8 **v21; // rax
  unsigned __int8 **v22; // rdx
  unsigned __int8 **result; // rax
  unsigned __int8 *v24; // rax
  int v25; // edx
  int v26; // edi
  unsigned __int8 *v27; // rdx
  unsigned __int8 **v28; // rax
  _QWORD *v29; // [rsp+0h] [rbp-80h]
  __int64 v30; // [rsp+8h] [rbp-78h]
  unsigned __int8 **v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+40h] [rbp-40h] BYREF
  int v33; // [rsp+48h] [rbp-38h]

  v6 = a4;
  v8 = 2LL * a4;
  v10 = (__int64 *)&a1[5][v8];
  if ( *(_DWORD *)(*v10 + 24) == 156 )
  {
    v24 = sub_3774B80((unsigned int *)a1[6], v10, v10, (__int64)a2, (__int64)a2, a5);
    v26 = v25;
    v27 = v24;
    v28 = *a1;
    *v28 = v27;
    *((_DWORD *)v28 + 2) = v26;
  }
  else
  {
    v11 = a1[2];
    v13 = (__int64)a1[3];
    v31 = a1[1];
    v32 = 0;
    v33 = 0;
    v14 = sub_33F17F0(v31, 51, (__int64)&v32, (unsigned int)v11, v13);
    v16 = (__int64)v31;
    if ( v32 )
    {
      v29 = v14;
      v30 = v15;
      sub_B91220((__int64)&v32, v32);
      v14 = v29;
      v15 = v30;
      v16 = (__int64)v31;
    }
    v17 = (unsigned __int8 *)sub_33FCE10(
                               v16,
                               *((unsigned int *)a1 + 4),
                               (__int64)a1[3],
                               (__int64)a1[4],
                               (__int64)a1[5][2 * v6],
                               (__int64)a1[5][2 * v6 + 1],
                               a5,
                               (__int64)v14,
                               v15,
                               a2,
                               a3);
    v19 = v18;
    v20 = v17;
    v21 = *a1;
    *v21 = v20;
    *((_DWORD *)v21 + 2) = v19;
  }
  v22 = *a1;
  result = a1[5];
  result[v8] = **a1;
  LODWORD(result[v8 + 1]) = *((_DWORD *)v22 + 2);
  return result;
}
