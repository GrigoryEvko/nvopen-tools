// Function: sub_1F00BA0
// Address: 0x1f00ba0
//
__int64 __fastcall sub_1F00BA0(
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
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // r15
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // rax
  __int64 v21; // rcx
  unsigned int v22; // ecx
  _QWORD *v23; // r12
  double v24; // xmm4_8
  double v25; // xmm5_8
  unsigned __int8 v27; // [rsp+Fh] [rbp-41h]
  __int64 v28; // [rsp+18h] [rbp-38h]

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9D3C0 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_29;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9D3C0);
  v27 = 0;
  *(_QWORD *)(a1 + 160) = sub_14A4050(v13, a2);
  v28 = a2 + 72;
  v14 = *(_QWORD *)(a2 + 80);
  if ( v14 == a2 + 72 )
    return v27;
  do
  {
    while ( 1 )
    {
      v15 = v14;
      v14 = *(_QWORD *)(v14 + 8);
      v16 = *(_QWORD *)(v15 + 24);
      v17 = v15 + 16;
      if ( v16 != v15 + 16 )
        break;
LABEL_24:
      if ( v28 == v14 )
        return v27;
    }
    while ( 1 )
    {
      v20 = v16;
      v16 = *(_QWORD *)(v16 + 8);
      if ( *(_BYTE *)(v20 - 8) != 78 )
        goto LABEL_10;
      v21 = *(_QWORD *)(v20 - 48);
      if ( *(_BYTE *)(v21 + 16) || (*(_BYTE *)(v21 + 33) & 0x20) == 0 )
        goto LABEL_10;
      v22 = *(_DWORD *)(v21 + 36);
      v23 = (_QWORD *)(v20 - 24);
      if ( v22 != 130 )
        break;
      if ( !(unsigned __int8)sub_14A2BC0(*(_QWORD *)(a1 + 160)) )
      {
        sub_1EFB7F0((__int64)v23);
        goto LABEL_19;
      }
LABEL_10:
      if ( v16 == v17 )
        goto LABEL_24;
    }
    if ( v22 > 0x82 )
    {
      if ( v22 == 131 && !(unsigned __int8)sub_14A2B30(*(_QWORD *)(a1 + 160)) )
      {
        sub_1EFCAC0((__int64)v23);
        goto LABEL_19;
      }
      goto LABEL_10;
    }
    if ( v22 == 128 )
    {
      if ( !(unsigned __int8)sub_14A2B90(*(_QWORD *)(a1 + 160)) )
      {
        sub_1EFF2D0(v23, a3, a4, a5, a6, v24, v25, a9, a10);
        goto LABEL_19;
      }
      goto LABEL_10;
    }
    if ( v22 != 129 || (unsigned __int8)sub_14A2B60(*(_QWORD *)(a1 + 160)) )
      goto LABEL_10;
    sub_1EFDAC0(v23, a3, a4, a5, a6, v18, v19, a9, a10);
LABEL_19:
    v27 = 1;
    v14 = *(_QWORD *)(a2 + 80);
  }
  while ( v14 != v28 );
  return v27;
}
