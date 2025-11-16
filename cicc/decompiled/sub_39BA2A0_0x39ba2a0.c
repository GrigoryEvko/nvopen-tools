// Function: sub_39BA2A0
// Address: 0x39ba2a0
//
__int64 __fastcall sub_39BA2A0(__int64 a1, unsigned int a2, double a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rdi
  _BYTE *v14; // rax
  _QWORD v16[4]; // [rsp+10h] [rbp-70h] BYREF
  void *v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h]
  __int64 v19; // [rsp+40h] [rbp-40h]
  __int64 v20; // [rsp+48h] [rbp-38h]
  int v21; // [rsp+50h] [rbp-30h]
  __int64 v22; // [rsp+58h] [rbp-28h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v21 = 1;
  v20 = 0;
  v19 = 0;
  v18 = 0;
  v17 = &unk_49EFBE0;
  v22 = a1;
  if ( a3 == 0.0 )
  {
    v4 = sub_16E7EE0((__int64)&v17, " sched: [", 9u);
    v5 = sub_16E7A90(v4, a2);
    v6 = *(_QWORD *)(v5 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 2 )
    {
      sub_16E7EE0(v5, ":?]", 3u);
    }
    else
    {
      *(_BYTE *)(v6 + 2) = 93;
      *(_WORD *)v6 = 16186;
      *(_QWORD *)(v5 + 24) += 3LL;
    }
  }
  else
  {
    v7 = sub_16E7EE0((__int64)&v17, " sched: [", 9u);
    v8 = sub_16E7A90(v7, a2);
    v16[1] = ":%2.2f";
    *(double *)&v16[2] = a3;
    v16[0] = &unk_49E8778;
    v13 = sub_16E8450(v8, (__int64)v16, v9, v10, v11, v12);
    v14 = *(_BYTE **)(v13 + 24);
    if ( *(_BYTE **)(v13 + 16) == v14 )
    {
      sub_16E7EE0(v13, "]", 1u);
    }
    else
    {
      *v14 = 93;
      ++*(_QWORD *)(v13 + 24);
    }
  }
  if ( v20 != v18 )
    sub_16E7BA0((__int64 *)&v17);
  sub_16E7BC0((__int64 *)&v17);
  return a1;
}
