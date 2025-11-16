// Function: sub_14835F0
// Address: 0x14835f0
//
__int64 __fastcall sub_14835F0(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, __m128i a5, __m128i a6)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 result; // rax
  __int64 v9; // rax
  _QWORD *v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r8
  unsigned int v17; // eax
  __int64 *v18; // r14
  unsigned int v19; // r13d
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r13
  __int64 v24; // rax
  _QWORD *v25; // r9
  __int64 v26; // rdx
  int v27; // [rsp+40h] [rbp-120h]
  _QWORD *v30; // [rsp+48h] [rbp-118h]
  unsigned int v31; // [rsp+50h] [rbp-110h]
  __int64 *v32; // [rsp+50h] [rbp-110h]
  __int64 v33; // [rsp+58h] [rbp-108h]
  __int64 v34; // [rsp+58h] [rbp-108h]
  __int64 v35; // [rsp+58h] [rbp-108h]
  __int64 v36; // [rsp+58h] [rbp-108h]
  __int64 v37; // [rsp+58h] [rbp-108h]
  __int64 v38; // [rsp+60h] [rbp-100h] BYREF
  __int64 v39; // [rsp+68h] [rbp-F8h] BYREF
  __int64 *v40; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v41; // [rsp+78h] [rbp-E8h]
  _BYTE v42[32]; // [rsp+80h] [rbp-E0h] BYREF
  unsigned __int64 v43[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE v44[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v6 = sub_1456E10((__int64)a1, a3);
  v43[0] = (unsigned __int64)v44;
  v7 = v6;
  v43[1] = 0x2000000000LL;
  sub_16BD3E0(v43, 1);
  sub_16BD4C0(v43, a2);
  sub_16BD4C0(v43, v7);
  v38 = 0;
  result = sub_16BDDE0(a1 + 102, v43, &v38);
  if ( result )
    goto LABEL_2;
  v9 = *(unsigned __int16 *)(a2 + 24);
  v10 = a1 + 102;
  switch ( (_WORD)v9 )
  {
    case 0:
      v11 = sub_15A43B0(*(_QWORD *)(a2 + 32), v7, 0);
      result = sub_145CE20((__int64)a1, v11);
      goto LABEL_2;
    case 1:
      result = sub_14835F0(a1, *(_QWORD *)(a2 + 32), v7, a4 + 1);
      goto LABEL_2;
    case 3:
      result = sub_1483BD0(a1, *(_QWORD *)(a2 + 32), v7);
      goto LABEL_2;
    case 2:
      result = sub_1483B20(a1, *(_QWORD *)(a2 + 32), v7);
      goto LABEL_2;
  }
  if ( a4 > dword_4F9B180 )
    goto LABEL_33;
  if ( (unsigned __int16)(v9 - 4) <= 1u )
  {
    v12 = *(_QWORD *)(a2 + 40);
    v40 = (__int64 *)v42;
    v41 = 0x400000000LL;
    v27 = v12;
    if ( (_DWORD)v12 )
    {
      v31 = 0;
      v13 = 0;
      do
      {
        v14 = sub_14835F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * v13), v7, a4 + 1);
        v15 = *(_QWORD *)(a2 + 32);
        v39 = v14;
        if ( (unsigned __int16)(*(_WORD *)(*(_QWORD *)(v15 + 8LL * v13) + 24LL) - 1) > 2u )
          v31 += *(_WORD *)(v14 + 24) == 1;
        ++v13;
        sub_1458920((__int64)&v40, &v39);
      }
      while ( v27 != v13 && v31 <= 1 );
      if ( v31 == 2 )
      {
        result = sub_16BDDE0(a1 + 102, v43, &v38);
        if ( result )
          goto LABEL_23;
        v10 = a1 + 102;
        if ( v40 != (__int64 *)v42 )
        {
          _libc_free((unsigned __int64)v40);
          v10 = a1 + 102;
        }
        LOWORD(v9) = *(_WORD *)(a2 + 24);
        goto LABEL_28;
      }
      LOWORD(v9) = *(_WORD *)(a2 + 24);
    }
    if ( (_WORD)v9 == 4 )
      result = (__int64)sub_147DD40((__int64)a1, (__int64 *)&v40, 0, 0, a5, a6);
    else
      result = sub_147EE30(a1, &v40, 0, 0, a5, a6);
LABEL_23:
    if ( v40 != (__int64 *)v42 )
    {
      v34 = result;
      _libc_free((unsigned __int64)v40);
      result = v34;
    }
    goto LABEL_2;
  }
LABEL_28:
  if ( (_WORD)v9 == 7 )
  {
    v16 = *(__int64 **)(a2 + 32);
    v40 = (__int64 *)v42;
    v41 = 0x400000000LL;
    v32 = &v16[*(_QWORD *)(a2 + 40)];
    if ( v16 == v32 )
    {
      result = sub_14785F0((__int64)a1, &v40, *(_QWORD *)(a2 + 48), 0);
    }
    else
    {
      v17 = a4;
      v18 = v16;
      v19 = v17 + 1;
      do
      {
        v20 = *v18++;
        v39 = sub_14835F0(a1, v20, v7, v19);
        sub_1458920((__int64)&v40, &v39);
      }
      while ( v32 != v18 );
      result = sub_14785F0((__int64)a1, &v40, *(_QWORD *)(a2 + 48), 0);
    }
    goto LABEL_23;
  }
LABEL_33:
  v30 = v10;
  v21 = sub_16BD760(v43, a1 + 108);
  v35 = v22;
  v23 = v21;
  v24 = sub_145CDC0(0x30u, a1 + 108);
  v25 = v30;
  if ( v24 )
  {
    v26 = v35;
    v36 = v24;
    sub_1456310(v24, v23, v26, a2, v7);
    v25 = v30;
    v24 = v36;
  }
  v37 = v24;
  sub_16BDA20(v25, v24, v38);
  sub_146DBF0((__int64)a1, v37);
  result = v37;
LABEL_2:
  if ( (_BYTE *)v43[0] != v44 )
  {
    v33 = result;
    _libc_free(v43[0]);
    return v33;
  }
  return result;
}
