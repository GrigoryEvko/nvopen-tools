// Function: sub_174B310
// Address: 0x174b310
//
__int64 __fastcall sub_174B310(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v4; // r10
  __int64 v5; // rbx
  __int64 v6; // r14
  unsigned int v7; // r12d
  unsigned int v8; // r13d
  char v9; // al
  __int64 v10; // r9
  char v11; // al
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // r15
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]
  __int64 v25; // [rsp+18h] [rbp-38h]
  __int64 v26; // [rsp+18h] [rbp-38h]

  v3 = a1;
  v4 = *(_QWORD *)a2;
  v5 = **(_QWORD **)(a2 - 24);
  v6 = *(_QWORD *)a3;
  v7 = *(unsigned __int8 *)(a2 + 16) - 24;
  v8 = *(unsigned __int8 *)(a3 + 16) - 24;
  v9 = *(_BYTE *)(v5 + 8);
  if ( v9 == 16 )
    v9 = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
  v10 = 0;
  if ( v9 == 15 )
  {
    v23 = *(_QWORD *)a2;
    v18 = sub_15A9650(*(_QWORD *)(a1 + 2664), **(_QWORD **)(a2 - 24));
    v4 = v23;
    v3 = a1;
    v10 = v18;
  }
  v11 = *(_BYTE *)(v4 + 8);
  if ( v11 == 16 )
    v11 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
  v12 = 0;
  if ( v11 == 15 )
  {
    v20 = v10;
    v22 = v3;
    v26 = v4;
    v17 = sub_15A9650(*(_QWORD *)(v3 + 2664), v4);
    v10 = v20;
    v3 = v22;
    v4 = v26;
    v12 = v17;
  }
  v13 = *(_BYTE *)(v6 + 8);
  if ( v13 == 16 )
    v13 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  v14 = 0;
  if ( v13 == 15 )
  {
    v19 = v4;
    v21 = v12;
    v25 = v10;
    v16 = sub_15A9650(*(_QWORD *)(v3 + 2664), v6);
    v4 = v19;
    v12 = v21;
    v10 = v25;
    v14 = v16;
  }
  v24 = v10;
  result = sub_15FB960(v7, v8, v5, v4, v6, v10, v12, v14);
  if ( (_DWORD)result == 46 && v14 != v5 )
    return 0;
  if ( (_DWORD)result == 45 && v24 != v6 )
    return 0;
  return result;
}
