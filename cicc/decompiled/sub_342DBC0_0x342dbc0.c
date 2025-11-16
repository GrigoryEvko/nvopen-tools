// Function: sub_342DBC0
// Address: 0x342dbc0
//
void __fastcall sub_342DBC0(__int64 a1)
{
  bool v2; // zf
  __int64 v3; // rax
  char v4; // al
  char v5; // al
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+10h] [rbp-80h]
  char v26; // [rsp+18h] [rbp-78h]
  char v27; // [rsp+18h] [rbp-78h]
  _QWORD *v28; // [rsp+18h] [rbp-78h]
  int v29[2]; // [rsp+38h] [rbp-58h] BYREF
  _QWORD *v30; // [rsp+40h] [rbp-50h]
  __int64 v31; // [rsp+48h] [rbp-48h]
  _BYTE v32[64]; // [rsp+50h] [rbp-40h] BYREF

  v2 = (_BYTE)qword_5039B68 == 0;
  v30 = v32;
  v3 = *(_QWORD *)(a1 + 64);
  v31 = 0;
  v32[0] = 0;
  *(_BYTE *)(v3 + 762) = 0;
  if ( v2 )
  {
    sub_CA08F0(
      (__int64 *)v29,
      "combine1",
      8u,
      (__int64)"DAG Combining 1",
      15,
      byte_4F826E9[0],
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      36);
    v19 = 0;
    v20 = *(_QWORD *)(a1 + 64);
    if ( *(_BYTE *)(a1 + 760) )
      v19 = a1 + 80;
    sub_33230B0(v20, 0, v19, *(_DWORD *)(a1 + 792));
    if ( *(_QWORD *)v29 )
      sub_C9E2A0(*(__int64 *)v29);
  }
  sub_CA08F0(
    (__int64 *)v29,
    "legalize_types",
    0xEu,
    (__int64)"Type Legalization",
    17,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  v4 = sub_3762260(*(_QWORD *)(a1 + 64));
  if ( *(_QWORD *)v29 )
  {
    v26 = v4;
    sub_C9E2A0(*(__int64 *)v29);
    v4 = v26;
  }
  *(_BYTE *)(*(_QWORD *)(a1 + 64) + 762LL) = 1;
  if ( v4 && !(_BYTE)qword_5039B68 )
  {
    sub_CA08F0(
      (__int64 *)v29,
      "combine_lt",
      0xAu,
      (__int64)"DAG Combining after legalize types",
      34,
      byte_4F826E9[0],
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      36);
    v23 = 0;
    if ( *(_BYTE *)(a1 + 760) )
      v23 = a1 + 80;
    sub_33230B0(*(_QWORD *)(a1 + 64), 1u, v23, *(_DWORD *)(a1 + 792));
    if ( *(_QWORD *)v29 )
      sub_C9E2A0(*(__int64 *)v29);
  }
  sub_CA08F0(
    (__int64 *)v29,
    "legalize_vec",
    0xCu,
    (__int64)"Vector Legalization",
    19,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  v5 = sub_3773D20(*(_QWORD *)(a1 + 64));
  if ( *(_QWORD *)v29 )
  {
    v27 = v5;
    sub_C9E2A0(*(__int64 *)v29);
    v5 = v27;
  }
  if ( v5 )
  {
    sub_CA08F0(
      (__int64 *)v29,
      "legalize_types2",
      0xFu,
      (__int64)"Type Legalization 2",
      19,
      byte_4F826E9[0],
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      36);
    sub_3762260(*(_QWORD *)(a1 + 64));
    if ( *(_QWORD *)v29 )
      sub_C9E2A0(*(__int64 *)v29);
    if ( !(_BYTE)qword_5039B68 )
    {
      sub_CA08F0(
        (__int64 *)v29,
        "combine_lv",
        0xAu,
        (__int64)"DAG Combining after legalize vectors",
        36,
        byte_4F826E9[0],
        "sdag",
        4u,
        "Instruction Selection and Scheduling",
        36);
      v22 = 0;
      if ( *(_BYTE *)(a1 + 760) )
        v22 = a1 + 80;
      sub_33230B0(*(_QWORD *)(a1 + 64), 2u, v22, *(_DWORD *)(a1 + 792));
      if ( *(_QWORD *)v29 )
        sub_C9E2A0(*(__int64 *)v29);
    }
  }
  v6 = (__int64)"legalize";
  sub_CA08F0(
    (__int64 *)v29,
    "legalize",
    8u,
    (__int64)"DAG Legalization",
    16,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  sub_334C7C0(*(__int64 **)(a1 + 64));
  if ( *(_QWORD *)v29 )
    sub_C9E2A0(*(__int64 *)v29);
  if ( !(_BYTE)qword_5039B68 )
  {
    sub_CA08F0(
      (__int64 *)v29,
      "combine2",
      8u,
      (__int64)"DAG Combining 2",
      15,
      byte_4F826E9[0],
      "sdag",
      4u,
      "Instruction Selection and Scheduling",
      36);
    v21 = 0;
    if ( *(_BYTE *)(a1 + 760) )
      v21 = a1 + 80;
    v6 = 3;
    sub_33230B0(*(_QWORD *)(a1 + 64), 3u, v21, *(_DWORD *)(a1 + 792));
    if ( *(_QWORD *)v29 )
      sub_C9E2A0(*(__int64 *)v29);
  }
  if ( *(_DWORD *)(a1 + 792) )
    sub_342D400(a1, v6, v7, v8, v9, v10);
  sub_CA08F0(
    (__int64 *)v29,
    "isel",
    4u,
    (__int64)"Instruction Selection",
    21,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  sub_3421E80(a1, (__int64)"isel");
  if ( *(_QWORD *)v29 )
    sub_C9E2A0(*(__int64 *)v29);
  v28 = (_QWORD *)sub_3422200(a1);
  sub_CA08F0(
    (__int64 *)v29,
    "sched",
    5u,
    (__int64)"Instruction Scheduling",
    22,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  sub_335DD50(v28, *(_QWORD *)(a1 + 64), *(_QWORD *)(*(_QWORD *)(a1 + 24) + 744LL), v11, v12, v13);
  if ( *(_QWORD *)v29 )
    sub_C9E2A0(*(__int64 *)v29);
  v25 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 744LL);
  sub_CA08F0(
    (__int64 *)v29,
    "emit",
    4u,
    (__int64)"Instruction Creation",
    20,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  v14 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*v28 + 96LL))(v28, *(_QWORD *)(a1 + 24) + 752LL);
  *(_QWORD *)(*(_QWORD *)(a1 + 24) + 744LL) = v14;
  if ( *(_QWORD *)v29 )
  {
    v24 = v14;
    sub_C9E2A0(*(__int64 *)v29);
    v14 = v24;
  }
  if ( v14 != v25 )
    sub_3374490(*(_QWORD *)(a1 + 72), v25, v14);
  sub_CA08F0(
    (__int64 *)v29,
    "cleanup",
    7u,
    (__int64)"Instruction Scheduling Cleanup",
    30,
    byte_4F826E9[0],
    "sdag",
    4u,
    "Instruction Selection and Scheduling",
    36);
  (*(void (__fastcall **)(_QWORD *))(*v28 + 8LL))(v28);
  if ( *(_QWORD *)v29 )
    sub_C9E2A0(*(__int64 *)v29);
  sub_33CCE30(*(_QWORD *)(a1 + 64), (__int64)"cleanup", v15, v16, v17, v18);
  if ( v30 != (_QWORD *)v32 )
    j_j___libc_free_0((unsigned __int64)v30);
}
