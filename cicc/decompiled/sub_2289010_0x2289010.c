// Function: sub_2289010
// Address: 0x2289010
//
void __fastcall sub_2289010(__int64 a1, __int64 a2, char a3, void **a4)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 i; // rbx
  _QWORD *v7; // rsi
  __int64 v8; // rdi
  _WORD *v9; // rdx
  __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h]
  char v12; // [rsp+10h] [rbp-50h]
  char v13; // [rsp+11h] [rbp-4Fh]
  __m128i *v14[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15; // [rsp+30h] [rbp-30h] BYREF

  v10 = a1;
  v11 = a2;
  v13 = a3;
  v12 = 0;
  sub_CA0F50((__int64 *)v14, a4);
  sub_2286660(&v10, v14);
  v4 = *(_QWORD *)(*(_QWORD *)v11 + 8LL);
  v5 = *(_QWORD *)(v4 + 32);
  for ( i = v4 + 16; i != v5; v5 = sub_220EF30(v5) )
  {
    v7 = *(_QWORD **)(v5 + 40);
    if ( (_BYTE)qword_4FDB148 || v7[1] )
      sub_2287970(&v10, v7);
  }
  v8 = v10;
  v9 = *(_WORD **)(v10 + 32);
  if ( *(_QWORD *)(v10 + 24) - (_QWORD)v9 <= 1u )
  {
    sub_CB6200(v10, "}\n", 2u);
  }
  else
  {
    *v9 = 2685;
    *(_QWORD *)(v8 + 32) += 2LL;
  }
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0((unsigned __int64)v14[0]);
}
