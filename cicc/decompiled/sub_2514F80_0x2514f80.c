// Function: sub_2514F80
// Address: 0x2514f80
//
void __fastcall sub_2514F80(__int64 a1, __int64 a2, char a3, void **a4)
{
  __int64 *v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // rsi
  __int64 v7; // rdi
  _WORD *v8; // rdx
  __int64 v9; // [rsp+0h] [rbp-60h] BYREF
  __int64 v10; // [rsp+8h] [rbp-58h]
  char v11; // [rsp+10h] [rbp-50h]
  char v12; // [rsp+11h] [rbp-4Fh]
  __int64 *v13[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+30h] [rbp-30h] BYREF

  v9 = a1;
  v10 = a2;
  v12 = a3;
  v11 = 0;
  sub_CA0F50((__int64 *)v13, a4);
  sub_25110D0(&v9, v13);
  v4 = *(__int64 **)(*(_QWORD *)v10 + 40LL);
  v5 = &v4[*(unsigned int *)(*(_QWORD *)v10 + 48LL)];
  while ( v5 != v4 )
  {
    v6 = *v4++;
    sub_25146E0((__int64)&v9, v6 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v7 = v9;
  v8 = *(_WORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v8 <= 1u )
  {
    sub_CB6200(v9, "}\n", 2u);
  }
  else
  {
    *v8 = 2685;
    *(_QWORD *)(v7 + 32) += 2LL;
  }
  if ( v13[0] != &v14 )
    j_j___libc_free_0((unsigned __int64)v13[0]);
}
