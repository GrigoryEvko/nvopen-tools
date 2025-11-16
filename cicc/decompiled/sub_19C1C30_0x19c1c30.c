// Function: sub_19C1C30
// Address: 0x19c1c30
//
char __fastcall sub_19C1C30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  unsigned int v15; // eax
  __int64 v16; // rbx
  _QWORD *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  _BYTE *v21; // rsi
  __int64 i; // r13
  _BYTE *v23; // rsi
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  char result; // al
  __int64 v28; // [rsp+0h] [rbp-60h]
  _QWORD *v29; // [rsp+8h] [rbp-58h]
  _QWORD v32[7]; // [rsp+28h] [rbp-38h] BYREF

  v15 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v15 )
  {
    v16 = 0;
    v17 = v32;
    v18 = 24LL * v15;
    do
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v19 = *(_QWORD *)(a1 - 8);
      else
        v19 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v20 = *(_QWORD *)(v19 + v16);
      if ( *(_BYTE *)(v20 + 16) > 0x17u )
      {
        v32[0] = v20;
        v21 = *(_BYTE **)(a3 + 8);
        if ( v21 == *(_BYTE **)(a3 + 16) )
        {
          v28 = v18;
          v29 = v17;
          sub_170B610(a3, v21, v17);
          v18 = v28;
          v17 = v29;
        }
        else
        {
          if ( v21 )
          {
            *(_QWORD *)v21 = v20;
            v21 = *(_BYTE **)(a3 + 8);
          }
          *(_QWORD *)(a3 + 8) = v21 + 8;
        }
      }
      v16 += 24;
    }
    while ( v16 != v18 );
  }
  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v24 = sub_1648700(i);
      v23 = *(_BYTE **)(a3 + 8);
      v32[0] = v24;
      if ( v23 != *(_BYTE **)(a3 + 16) )
        break;
      sub_17C2330(a3, v23, v32);
      i = *(_QWORD *)(i + 8);
      if ( !i )
        goto LABEL_19;
    }
    if ( v23 )
    {
      *(_QWORD *)v23 = v24;
      v23 = *(_BYTE **)(a3 + 8);
    }
    *(_QWORD *)(a3 + 8) = v23 + 8;
  }
LABEL_19:
  sub_14045C0(a5, a1, a4);
  sub_19C03B0(a1, (char **)a3);
  sub_164D160(a1, a2, a6, a7, a8, a9, v25, v26, a12, a13);
  result = sub_15F3040(a1);
  if ( !result )
  {
    result = sub_15F3330(a1);
    if ( !result )
      return sub_15F20C0((_QWORD *)a1);
  }
  return result;
}
