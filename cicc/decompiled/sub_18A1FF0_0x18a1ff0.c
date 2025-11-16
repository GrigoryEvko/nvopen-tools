// Function: sub_18A1FF0
// Address: 0x18a1ff0
//
__int64 __fastcall sub_18A1FF0(
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
  _QWORD *v10; // r8
  _QWORD *v12; // rax
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // r13
  _QWORD *v18; // r12
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r15
  unsigned __int8 v24; // al
  unsigned __int64 v25; // rax
  __int64 v26; // rsi
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 result; // rax
  unsigned __int64 v30; // rax
  int v31; // ebx
  unsigned __int64 v32; // r12
  _QWORD *v33; // r13
  unsigned int i; // r15d
  __int64 v35; // rax
  __int64 *v36; // rbx
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // [rsp+8h] [rbp-48h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  __int64 v42; // [rsp+18h] [rbp-38h]
  __int64 v43; // [rsp+18h] [rbp-38h]

  v10 = (_QWORD *)(a2 + 16);
  v12 = *(_QWORD **)(a2 + 24);
  v13 = *(_QWORD *)(a1 + 56);
  if ( v12 )
  {
    v14 = v10;
    do
    {
      while ( 1 )
      {
        v15 = v12[2];
        v16 = v12[3];
        if ( v12[4] >= v13 )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v16 )
          goto LABEL_6;
      }
      v14 = v12;
      v12 = (_QWORD *)v12[2];
    }
    while ( v15 );
LABEL_6:
    if ( v10 != v14 && v14[4] <= v13 )
      v10 = v14;
  }
  v17 = *(_QWORD *)(a1 + 48);
  v18 = (_QWORD *)(a1 + 40);
  v40 = v10[5];
  if ( a1 + 40 == v17 )
  {
LABEL_30:
    v30 = sub_157EBA0(a1);
    if ( !v30 )
      return sub_157F980(a1);
    v31 = sub_15F4D60(v30);
    v32 = sub_157EBA0(a1);
    if ( (unsigned __int64)v31 > 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v41 = 8LL * v31;
    if ( !v31 )
      return sub_157F980(a1);
    v43 = sub_22077B0(8LL * v31);
    v33 = (_QWORD *)v43;
    for ( i = 0; i != v31; ++i )
    {
      v35 = sub_15F4DF0(v32, i);
      if ( v33 )
        *v33 = v35;
      ++v33;
    }
    if ( i )
    {
      v36 = (__int64 *)v43;
      do
      {
        v37 = *v36++;
        sub_157F2D0(v37, a1, 0);
      }
      while ( (__int64 *)(v43 + 8LL * (i - 1) + 8) != v36 );
    }
    sub_157F980(a1);
    result = v43;
    if ( v43 )
      return j_j___libc_free_0(v43, v41);
    return result;
  }
  while ( 1 )
  {
    v22 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
    v18 = (_QWORD *)v22;
    if ( !v22 )
      BUG();
    v23 = v22 - 24;
    if ( *(_BYTE *)(*(_QWORD *)(v22 - 24) + 8LL) == 10 )
      break;
    v24 = *(_BYTE *)(v22 - 8);
    if ( v24 > 0x17u )
    {
      if ( v24 == 78 )
      {
        v25 = v22 - 24;
        v26 = v23 | 4;
LABEL_20:
        v27 = v25 - 24;
        v28 = v25 - 72;
        if ( (v26 & 4) != 0 )
          v28 = v27;
        if ( *(_BYTE *)(*(_QWORD *)v28 + 16LL)
          || (v42 = *(_QWORD *)v28, !sub_15E1830(*(_DWORD *)(*(_QWORD *)v28 + 36LL)))
          || (*(_BYTE *)(v42 + 33) & 0x20) == 0 )
        {
          sub_13983A0(v40, v26);
        }
        goto LABEL_12;
      }
      if ( v24 == 29 )
      {
        v26 = v22 - 24;
        v25 = v23 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v23 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_20;
      }
    }
LABEL_12:
    if ( *(_QWORD *)(v22 - 16) )
    {
      v19 = sub_1599EF0(*(__int64 ***)(v22 - 24));
      sub_164D160(v22 - 24, v19, a3, a4, a5, a6, v20, v21, a9, a10);
    }
    if ( v22 == v17 )
      goto LABEL_30;
  }
  result = (unsigned int)*(unsigned __int8 *)(v22 - 8) - 25;
  if ( (unsigned int)result > 9 )
  {
    v38 = *(_QWORD *)(v22 + 8);
    if ( v38 == *(_QWORD *)(v22 + 16) + 40LL || !v38 )
      v39 = 0;
    else
      v39 = v38 - 24;
    return sub_1AEE6A0(v39, 0, 0, 0);
  }
  return result;
}
