// Function: sub_139B410
// Address: 0x139b410
//
__int64 __fastcall sub_139B410(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 i; // rbx
  __int64 v7; // rdi
  _WORD *v8; // rdx
  __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  char v12; // [rsp+10h] [rbp-50h]
  __int64 v13[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v14; // [rsp+30h] [rbp-30h] BYREF

  v10 = a1;
  v11 = a2;
  v12 = a3;
  sub_16E2FC0(v13, a4);
  sub_139A010(&v10, v13);
  v5 = *(_QWORD *)(*(_QWORD *)v11 + 32LL);
  for ( i = *(_QWORD *)v11 + 16LL; i != v5; v5 = sub_220EEE0(v5) )
    sub_139A6D0(&v10, *(_QWORD **)(v5 + 40));
  v7 = v10;
  v8 = *(_WORD **)(v10 + 24);
  if ( *(_QWORD *)(v10 + 16) - (_QWORD)v8 <= 1u )
  {
    sub_16E7EE0(v10, "}\n", 2);
  }
  else
  {
    *v8 = 2685;
    *(_QWORD *)(v7 + 24) += 2LL;
  }
  if ( (__int64 *)v13[0] != &v14 )
    j_j___libc_free_0(v13[0], v14 + 1);
  return a1;
}
