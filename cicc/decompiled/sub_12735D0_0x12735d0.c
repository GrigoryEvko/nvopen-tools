// Function: sub_12735D0
// Address: 0x12735d0
//
__int64 __fastcall sub_12735D0(_QWORD *a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 i; // rax
  __int64 v6; // r15
  __int64 j; // rax
  _QWORD *v8; // r15
  int v9; // ebx
  __int64 v10; // r14
  __int64 result; // rax
  unsigned __int32 *v12; // rbx
  signed __int32 v13; // ecx
  signed __int32 v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int32 v18; // r15d
  __int64 v20; // [rsp+0h] [rbp-40h]

  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = *(_QWORD *)(i + 160);
  if ( !sub_8D23B0(v6) )
    sub_1273520(a1, a4, v6, 0);
  for ( j = *(_QWORD *)(a2 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v8 = **(_QWORD ***)(j + 168);
  if ( v8 )
  {
    v9 = 1;
    do
    {
      v10 = v8[1];
      if ( !sub_8D23B0(v10) )
        sub_1273520(a1, a4, v10, v9);
      v8 = (_QWORD *)*v8;
      ++v9;
    }
    while ( v8 );
  }
  if ( (*(_BYTE *)(a2 + 198) & 0x20) != 0 )
  {
    sub_1273260(a1, a4, (__int64)"kernel", 1u);
    v15 = *(_QWORD *)(a3 + 16);
    v16 = v15 + 40;
    v17 = v15 + 8 * (5LL * *(unsigned int *)(a3 + 8) + 5);
    if ( v17 != v16 )
    {
      v18 = 1;
      do
      {
        if ( *(_BYTE *)(v16 + 33) )
        {
          v20 = v17;
          sub_1273260(a1, a4, (__int64)"grid_constant", v18);
          v17 = v20;
        }
        v16 += 40;
        ++v18;
      }
      while ( v16 != v17 );
    }
  }
  if ( (*(_BYTE *)(a2 + 199) & 4) != 0 )
    sub_1273260(a1, a4, (__int64)"full_custom_abi", 0xFFFFFFFF);
  result = a2;
  v12 = *(unsigned __int32 **)(a2 + 336);
  if ( v12 )
  {
    if ( (*v12 & 0x80000000) == 0 )
    {
      result = sub_1273260(a1, a4, (__int64)"preserve_n_data", *v12);
      v13 = v12[1];
      *v12 = -1;
      if ( v13 < 0 )
      {
LABEL_18:
        v14 = v12[2];
        if ( v14 < 0 )
          return result;
LABEL_20:
        result = sub_1273260(a1, a4, (__int64)"preserve_n_after", v14);
        v12[2] = -1;
        return result;
      }
    }
    else
    {
      v13 = v12[1];
      if ( v13 < 0 )
        goto LABEL_18;
    }
    result = sub_1273260(a1, a4, (__int64)"preserve_n_control", v13);
    v14 = v12[2];
    v12[1] = -1;
    if ( v14 >= 0 )
      goto LABEL_20;
  }
  return result;
}
