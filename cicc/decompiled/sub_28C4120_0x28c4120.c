// Function: sub_28C4120
// Address: 0x28c4120
//
unsigned __int8 *__fastcall sub_28C4120(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 **v8; // r15
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  __int64 v11; // r8
  _QWORD *v12; // rax
  int v13; // edx
  _BYTE *v14; // rcx
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // r15
  unsigned __int8 **v20; // r15
  int v21; // r14d
  __int64 v22; // rax
  int v23; // ebx
  signed __int64 v24; // rax
  unsigned int v25; // r9d
  unsigned __int8 *result; // rax
  __int64 v27; // rcx
  unsigned __int8 v28; // dl
  int v29; // edx
  unsigned __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned __int64 v36; // [rsp+0h] [rbp-70h]
  unsigned __int64 v37; // [rsp+8h] [rbp-68h]
  unsigned __int64 v38; // [rsp+8h] [rbp-68h]
  unsigned __int64 v39; // [rsp+8h] [rbp-68h]
  int v40; // [rsp+8h] [rbp-68h]
  _BYTE *v41; // [rsp+10h] [rbp-60h] BYREF
  __int64 v42; // [rsp+18h] [rbp-58h]
  _BYTE v43[80]; // [rsp+20h] [rbp-50h] BYREF

  v7 = *(_DWORD *)(a2 + 4);
  v8 = (__int64 **)a1[5];
  v41 = v43;
  v9 = v7 & 0x7FFFFFF;
  v10 = (_QWORD *)(a2 + 32 * (1 - v9));
  v42 = 0x400000000LL;
  v11 = (-32 * (1 - v9)) >> 5;
  if ( (unsigned __int64)(-32 * (1 - v9)) > 0x80 )
  {
    v40 = (-32 * (1 - v9)) >> 5;
    sub_C8D5F0((__int64)&v41, v43, (-32 * (1 - v9)) >> 5, 8u, v11, a6);
    v14 = v41;
    v13 = v42;
    LODWORD(v11) = v40;
    v12 = &v41[8 * (unsigned int)v42];
  }
  else
  {
    v12 = v43;
    v13 = 0;
    v14 = v43;
  }
  if ( (_QWORD *)a2 != v10 )
  {
    do
    {
      if ( v12 )
        *v12 = *v10;
      v10 += 4;
      ++v12;
    }
    while ( (_QWORD *)a2 != v10 );
    v14 = v41;
    v13 = v42;
  }
  v15 = *(_DWORD *)(a2 + 4);
  v16 = *(_QWORD *)(a2 + 72);
  LODWORD(v42) = v13 + v11;
  v17 = sub_DF9500(v8, v16, *(_QWORD *)(a2 - 32LL * (v15 & 0x7FFFFFF)), (__int64)v14, (unsigned int)(v13 + v11));
  if ( v18 || v17 )
  {
    if ( v41 != v43 )
      _libc_free((unsigned __int64)v41);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v19 = *(_QWORD *)(a2 - 8);
    else
      v19 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v20 = (unsigned __int8 **)(v19 + 32);
    v21 = 2;
    v22 = sub_BB5290(a2);
    v23 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v24 = v22 & 0xFFFFFFFFFFFFFFF9LL | 4;
    if ( v23 == 1 )
      return 0;
    while ( 1 )
    {
      v30 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = (v24 >> 1) & 3;
      if ( ((v24 >> 1) & 3) == 0 )
      {
        v27 = sub_BCBAE0(v24 & 0xFFFFFFFFFFFFFFF8LL, *v20, v31);
        goto LABEL_25;
      }
      v25 = v21 - 2;
      if ( !v24 )
        break;
      if ( v31 == 2 )
      {
        if ( v30 )
        {
          v37 = v24 & 0xFFFFFFFFFFFFFFF8LL;
          result = sub_28C3FC0(a1, a2, v25, v24 & 0xFFFFFFFFFFFFFFF8LL);
          v27 = v37;
          if ( result )
            return result;
          goto LABEL_19;
        }
      }
      else
      {
        if ( (_DWORD)v31 != 1 )
          break;
        if ( v30 )
        {
          v38 = v24 & 0xFFFFFFFFFFFFFFF8LL;
          result = sub_28C3FC0(a1, a2, v25, *(_QWORD *)(v30 + 24));
          if ( result )
            return result;
          v27 = *(_QWORD *)(v38 + 24);
LABEL_19:
          v28 = *(_BYTE *)(v27 + 8);
          if ( v28 == 16 )
            goto LABEL_20;
          goto LABEL_26;
        }
      }
      v36 = v24 & 0xFFFFFFFFFFFFFFF8LL;
      v34 = sub_BCBAE0(0, *v20, v31);
      result = sub_28C3FC0(a1, a2, v21 - 2, v34);
      if ( result )
        return result;
      v27 = sub_BCBAE0(v36, *v20, v35);
LABEL_25:
      v28 = *(_BYTE *)(v27 + 8);
      if ( v28 == 16 )
      {
LABEL_20:
        v24 = *(_QWORD *)(v27 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
        goto LABEL_21;
      }
LABEL_26:
      if ( (unsigned int)v28 - 17 > 1 )
      {
        v24 = v27 & 0xFFFFFFFFFFFFFFF9LL;
        if ( v28 != 15 )
          v24 = 0;
LABEL_21:
        v20 += 4;
        v29 = v21 + 1;
        if ( v23 == v21 )
          return 0;
        goto LABEL_22;
      }
      v20 += 4;
      v29 = v21 + 1;
      v24 = v27 & 0xFFFFFFFFFFFFFFF9LL | 2;
      if ( v23 == v21 )
        return 0;
LABEL_22:
      v21 = v29;
    }
    v39 = v24 & 0xFFFFFFFFFFFFFFF8LL;
    v32 = sub_BCBAE0(v24 & 0xFFFFFFFFFFFFFFF8LL, *v20, v31);
    result = sub_28C3FC0(a1, a2, v21 - 2, v32);
    if ( result )
      return result;
    v27 = sub_BCBAE0(v39, *v20, v33);
    goto LABEL_25;
  }
  if ( v41 != v43 )
  {
    _libc_free((unsigned __int64)v41);
    return 0;
  }
  return 0;
}
