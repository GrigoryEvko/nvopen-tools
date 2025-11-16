// Function: sub_1D50350
// Address: 0x1d50350
//
void __fastcall sub_1D50350(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  bool v5; // zf
  char v6; // al
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // r9d
  __int64 v14; // [rsp+0h] [rbp-90h]
  __int64 v15; // [rsp+0h] [rbp-90h]
  __int64 v16; // [rsp+10h] [rbp-80h]
  char v17; // [rsp+18h] [rbp-78h]
  char v18; // [rsp+18h] [rbp-78h]
  _QWORD *v19; // [rsp+18h] [rbp-78h]
  int v20[2]; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v21; // [rsp+40h] [rbp-50h]
  __int64 v22; // [rsp+48h] [rbp-48h]
  _QWORD v23[8]; // [rsp+50h] [rbp-40h] BYREF

  v1 = *(__int64 **)(a1 + 8);
  v22 = 0;
  LOBYTE(v23[0]) = 0;
  v21 = v23;
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_42:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9D3C0 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_42;
  }
  v4 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9D3C0);
  sub_14A4050(v4, **(_QWORD **)(a1 + 248));
  v5 = byte_4FC1BE0 == 0;
  *(_BYTE *)(*(_QWORD *)(a1 + 272) + 658LL) = 0;
  if ( v5 )
  {
    sub_16D8B50(
      (__m128i **)v20,
      "combine1",
      8u,
      (__int64)"DAG Combining 1",
      15,
      unk_4F9E388,
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      (double *)0x24);
    sub_1FD1B90(*(_QWORD *)(a1 + 272), 0, *(_QWORD *)(a1 + 288), *(unsigned int *)(a1 + 304));
    if ( *(_QWORD *)v20 )
      sub_16D7950(*(__int64 *)v20);
  }
  sub_16D8B50(
    (__m128i **)v20,
    "legalize_types",
    0xEu,
    (__int64)"Type Legalization",
    17,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  v6 = sub_20145E0(*(_QWORD *)(a1 + 272));
  if ( *(_QWORD *)v20 )
  {
    v17 = v6;
    sub_16D7950(*(__int64 *)v20);
    v6 = v17;
  }
  *(_BYTE *)(*(_QWORD *)(a1 + 272) + 658LL) = 1;
  if ( v6 )
  {
    if ( !byte_4FC1BE0 )
    {
      sub_16D8B50(
        (__m128i **)v20,
        "combine_lt",
        0xAu,
        (__int64)"DAG Combining after legalize types",
        34,
        unk_4F9E388,
        "sdag",
        4u,
        "Instruction Selection and Scheduling",
        (double *)0x24);
      sub_1FD1B90(*(_QWORD *)(a1 + 272), 1, *(_QWORD *)(a1 + 288), *(unsigned int *)(a1 + 304));
      if ( *(_QWORD *)v20 )
        sub_16D7950(*(__int64 *)v20);
    }
  }
  sub_16D8B50(
    (__m128i **)v20,
    "legalize_vec",
    0xCu,
    (__int64)"Vector Legalization",
    19,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  v7 = sub_20211B0(*(_QWORD *)(a1 + 272));
  if ( *(_QWORD *)v20 )
  {
    v18 = v7;
    sub_16D7950(*(__int64 *)v20);
    v7 = v18;
  }
  if ( v7 )
  {
    sub_16D8B50(
      (__m128i **)v20,
      "legalize_types2",
      0xFu,
      (__int64)"Type Legalization 2",
      19,
      unk_4F9E388,
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      (double *)0x24);
    sub_20145E0(*(_QWORD *)(a1 + 272));
    if ( *(_QWORD *)v20 )
      sub_16D7950(*(__int64 *)v20);
    if ( !byte_4FC1BE0 )
    {
      sub_16D8B50(
        (__m128i **)v20,
        "combine_lv",
        0xAu,
        (__int64)"DAG Combining after legalize vectors",
        36,
        unk_4F9E388,
        "sdag",
        4u,
        "Instruction Selection and Scheduling",
        (double *)0x24);
      sub_1FD1B90(*(_QWORD *)(a1 + 272), 2, *(_QWORD *)(a1 + 288), *(unsigned int *)(a1 + 304));
      if ( *(_QWORD *)v20 )
        sub_16D7950(*(__int64 *)v20);
    }
  }
  sub_16D8B50(
    (__m128i **)v20,
    "legalize",
    8u,
    (__int64)"DAG Legalization",
    16,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  sub_1FFFE20(*(_QWORD *)(a1 + 272));
  if ( *(_QWORD *)v20 )
    sub_16D7950(*(__int64 *)v20);
  if ( !byte_4FC1BE0 )
  {
    sub_16D8B50(
      (__m128i **)v20,
      "combine2",
      8u,
      (__int64)"DAG Combining 2",
      15,
      unk_4F9E388,
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      (double *)0x24);
    sub_1FD1B90(*(_QWORD *)(a1 + 272), 3, *(_QWORD *)(a1 + 288), *(unsigned int *)(a1 + 304));
    if ( *(_QWORD *)v20 )
      sub_16D7950(*(__int64 *)v20);
  }
  if ( *(_DWORD *)(a1 + 304) )
    sub_1D4FC30(a1);
  sub_16D8B50(
    (__m128i **)v20,
    (unsigned __int8 *)"isel",
    4u,
    (__int64)"Instruction Selection",
    21,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  sub_1D490E0(a1);
  if ( *(_QWORD *)v20 )
    sub_16D7950(*(__int64 *)v20);
  v19 = (_QWORD *)sub_1D493E0(a1);
  sub_16D8B50(
    (__m128i **)v20,
    (unsigned __int8 *)"sched",
    5u,
    (__int64)"Instruction Scheduling",
    22,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  sub_1D0DAC0(v19, *(_QWORD *)(a1 + 272), *(_QWORD *)(*(_QWORD *)(a1 + 248) + 784LL));
  if ( *(_QWORD *)v20 )
    sub_16D7950(*(__int64 *)v20);
  v16 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 784LL);
  sub_16D8B50(
    (__m128i **)v20,
    (unsigned __int8 *)"emit",
    4u,
    (__int64)"Instruction Creation",
    20,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  v14 = *(_QWORD *)(a1 + 248);
  v8 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v19 + 88LL))(v19, v14 + 792);
  v9 = v8;
  *(_QWORD *)(v14 + 784) = v8;
  if ( *(_QWORD *)v20 )
  {
    v15 = v8;
    sub_16D7950(*(__int64 *)v20);
    v9 = v15;
  }
  if ( v9 != v16 )
    sub_2053A60(*(_QWORD *)(a1 + 280));
  sub_16D8B50(
    (__m128i **)v20,
    (unsigned __int8 *)"cleanup",
    7u,
    (__int64)"Instruction Scheduling Cleanup",
    30,
    unk_4F9E388,
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    (double *)0x24);
  (*(void (__fastcall **)(_QWORD *))(*v19 + 8LL))(v19);
  if ( *(_QWORD *)v20 )
    sub_16D7950(*(__int64 *)v20);
  sub_1D17A50(*(_QWORD *)(a1 + 272), (__int64)"cleanup", v10, v11, v12, v13);
  if ( v21 != v23 )
    j_j___libc_free_0(v21, v23[0] + 1LL);
}
