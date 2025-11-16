// Function: sub_10FFC90
// Address: 0x10ffc90
//
__int64 __fastcall sub_10FFC90(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v3; // r8
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rcx
  unsigned int v7; // r14d
  unsigned int v8; // r13d
  int v9; // edx
  __int64 v10; // r9
  int v11; // esi
  __int64 v12; // rdx
  int v13; // esi
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
  v4 = *((_QWORD *)a3 + 1);
  v5 = *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL);
  v6 = *((_QWORD *)a2 + 1);
  v7 = *a3 - 29;
  v8 = *a2 - 29;
  v9 = *(unsigned __int8 *)(v5 + 8);
  if ( (unsigned int)(v9 - 17) <= 1 )
    LOBYTE(v9) = *(_BYTE *)(**(_QWORD **)(v5 + 16) + 8LL);
  v10 = 0;
  if ( (_BYTE)v9 == 14 )
  {
    v23 = *((_QWORD *)a2 + 1);
    v18 = sub_AE4450(*(_QWORD *)(a1 + 88), *(_QWORD *)(*((_QWORD *)a2 - 4) + 8LL));
    v6 = v23;
    v3 = a1;
    v10 = v18;
  }
  v11 = *(unsigned __int8 *)(v6 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
    LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
  v12 = 0;
  if ( (_BYTE)v11 == 14 )
  {
    v20 = v10;
    v22 = v3;
    v26 = v6;
    v17 = sub_AE4450(*(_QWORD *)(v3 + 88), v6);
    v10 = v20;
    v3 = v22;
    v6 = v26;
    v12 = v17;
  }
  v13 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v13 - 17) <= 1 )
    LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
  v14 = 0;
  if ( (_BYTE)v13 == 14 )
  {
    v19 = v6;
    v21 = v12;
    v25 = v10;
    v16 = sub_AE4450(*(_QWORD *)(v3 + 88), v4);
    v6 = v19;
    v12 = v21;
    v10 = v25;
    v14 = v16;
  }
  v24 = v10;
  result = sub_B50810(v8, v7, v5, v6, v4, v10, v12, v14);
  if ( (_DWORD)result == 48 && v14 != v5 )
    return 0;
  if ( (_DWORD)result == 47 && v24 != v4 )
    return 0;
  return result;
}
