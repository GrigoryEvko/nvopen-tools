// Function: sub_1D37190
// Address: 0x1d37190
//
__int64 *__fastcall sub_1D37190(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        int a5,
        double a6,
        double a7,
        __m128i a8)
{
  unsigned __int64 v8; // rax
  __int64 v9; // r11
  unsigned __int64 v10; // r13
  unsigned __int8 *v11; // r15
  __int64 v12; // r9
  unsigned int v14; // esi
  __int64 v15; // r14
  unsigned __int64 v16; // r15
  unsigned int v17; // ebx
  __int64 v18; // rcx
  __int64 v19; // r13
  _BYTE *v20; // rax
  unsigned __int8 *v21; // rcx
  __int64 v22; // r10
  __int64 v23; // r8
  unsigned __int8 *v24; // rdx
  unsigned __int8 *v25; // rsi
  const void ***v26; // rax
  int v27; // edx
  __int64 *v28; // r12
  __int128 v30; // [rsp-10h] [rbp-D0h]
  __int64 v31; // [rsp+8h] [rbp-B8h]
  __int64 v32; // [rsp+10h] [rbp-B0h]
  _BYTE *v34; // [rsp+18h] [rbp-A8h]
  __int64 *v35; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v36; // [rsp+28h] [rbp-98h]
  unsigned __int64 v37; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v39; // [rsp+40h] [rbp-80h] BYREF
  __int64 v40; // [rsp+48h] [rbp-78h]
  _BYTE v41[112]; // [rsp+50h] [rbp-70h] BYREF

  v8 = a3;
  v9 = a2;
  v10 = a3;
  if ( a3 == 1 )
    return *(__int64 **)a2;
  v11 = v41;
  v12 = a1;
  v39 = v41;
  v40 = 0x400000000LL;
  if ( a3 > 4 )
  {
    v37 = a3;
    sub_16CD150((__int64)&v39, v41, a3, 16, a5, a1);
    a3 = (unsigned int)v40;
    v14 = HIDWORD(v40);
    v8 = v37;
    v9 = a2;
    v12 = a1;
LABEL_5:
    v15 = v9;
    v36 = v10;
    v16 = v8;
    v17 = 0;
    v18 = 0;
    v19 = v12;
    v20 = v41;
    while ( 1 )
    {
      v21 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v15 + 16 * v18) + 40LL)
                              + 16LL * *(unsigned int *)(v15 + 16 * v18 + 8));
      v22 = *((_QWORD *)v21 + 1);
      v23 = *v21;
      if ( (unsigned int)a3 >= v14 )
      {
        v31 = *v21;
        v32 = *((_QWORD *)v21 + 1);
        v34 = v20;
        sub_16CD150((__int64)&v39, v20, 0, 16, v23, v12);
        a3 = (unsigned int)v40;
        v23 = v31;
        v22 = v32;
        v20 = v34;
      }
      v24 = &v39[16 * a3];
      *(_QWORD *)v24 = v23;
      v18 = ++v17;
      *((_QWORD *)v24 + 1) = v22;
      a3 = (unsigned int)(v40 + 1);
      LODWORD(v40) = v40 + 1;
      if ( v17 >= v16 )
        break;
      v14 = HIDWORD(v40);
    }
    v12 = v19;
    v10 = v36;
    v11 = v20;
    v25 = v39;
    goto LABEL_11;
  }
  if ( a3 )
  {
    v14 = 4;
    a3 = 0;
    goto LABEL_5;
  }
  v25 = v41;
LABEL_11:
  v35 = (__int64 *)v12;
  v26 = (const void ***)sub_1D25C30(v12, v25, a3);
  *((_QWORD *)&v30 + 1) = v10;
  *(_QWORD *)&v30 = a2;
  v28 = sub_1D36D80(v35, 51, a4, v26, v27, a6, a7, a8, (__int64)v35, v30);
  if ( v39 != v11 )
    _libc_free((unsigned __int64)v39);
  return v28;
}
