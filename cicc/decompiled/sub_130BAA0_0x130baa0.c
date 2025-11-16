// Function: sub_130BAA0
// Address: 0x130baa0
//
__int64 __fastcall sub_130BAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // r10d
  __int64 v14; // r15
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  char v19; // [rsp-18h] [rbp-58h]
  char v20; // [rsp-18h] [rbp-58h]
  int v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+0h] [rbp-40h]
  __int64 v24; // [rsp+8h] [rbp-38h]
  __int64 v25; // [rsp+8h] [rbp-38h]
  int v26; // [rsp+8h] [rbp-38h]

  v9 = sub_131C0E0(*(_QWORD *)(a2 + 58376));
  if ( !*(_QWORD *)(*(_QWORD *)(v9 + 8) + 64LL) )
    return 1;
  v10 = a5 - a4;
  v19 = a6;
  v21 = a6;
  v24 = v9;
  v11 = sub_1345C00(a1, a2, v9, (int)a2 + 56, a3, v10, 4096, v19, 0);
  v12 = v24;
  v13 = v21;
  v14 = v11;
  if ( v11
    || (v20 = v21,
        v22 = v24,
        v26 = v13,
        v16 = sub_1345C00(a1, a2, v12, (int)a2 + 19496, a3, v10, 4096, v20, 0),
        v12 = v22,
        (v14 = v16) != 0) )
  {
    v25 = v12;
    if ( !(unsigned __int8)sub_13457B0(a1, a2, v12, a3, v14) )
      return 0;
    v18 = v25;
LABEL_9:
    sub_1344AD0(a1, a2, v18, v14);
    return 1;
  }
  v17 = sub_1345C30(a1, a2, v22, (int)a2 + 38936, a3, v10, 4096, v26, 0);
  v14 = v17;
  if ( !v17 )
    return 1;
  result = sub_13457B0(a1, a2, v22, a3, v17);
  v18 = v22;
  if ( (_BYTE)result )
    goto LABEL_9;
  if ( !v10 )
    return 0;
  _InterlockedAdd64((volatile signed __int64 *)(*(_QWORD *)(a2 + 62224) + 56LL), v10);
  return result;
}
