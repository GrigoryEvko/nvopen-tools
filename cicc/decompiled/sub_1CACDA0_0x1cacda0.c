// Function: sub_1CACDA0
// Address: 0x1cacda0
//
__int64 __fastcall sub_1CACDA0(
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
  __int64 *v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rcx
  double v20; // xmm4_8
  double v21; // xmm5_8
  bool v22; // zf
  char v23; // al
  __int16 v25; // [rsp+Eh] [rbp-22h] BYREF

  v10 = *(__int64 **)(a1 + 8);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_16:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9E06C )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_16;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9E06C);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13 + 160;
  v16 = *v14;
  v17 = v14[1];
  if ( v16 == v17 )
LABEL_15:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9D764 )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_15;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9D764);
  v19 = sub_14CF090(v18, a2);
  v22 = *(_BYTE *)(a1 + 153) == 0;
  LOBYTE(v25) = *(_BYTE *)(a1 + 154);
  v23 = 1;
  if ( !v22 )
    v23 = byte_4FBE540;
  HIBYTE(v25) = v23;
  return sub_1CACAA0(&v25, a2, v15, v19, a3, a4, a5, a6, v20, v21, a9, a10);
}
