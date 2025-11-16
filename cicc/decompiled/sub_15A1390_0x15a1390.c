// Function: sub_15A1390
// Address: 0x15a1390
//
__int64 __fastcall sub_15A1390(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v14; // r12
  __int64 v15; // rsi
  __int64 v16; // rbx
  _BYTE v17[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h] BYREF
  __int64 v19; // [rsp+10h] [rbp-30h]

  v5 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v5 == 16 )
    v5 = *(unsigned __int8 *)(**(_QWORD **)(a1 + 16) + 8LL);
  v6 = sub_1593350(v5, a2, a3, a4);
  v9 = sub_16982C0(v5, a2, v7, v8);
  v10 = v9;
  if ( v6 == v9 )
  {
    sub_169C580(&v18, v9, 0);
    if ( v18 != v10 )
      goto LABEL_5;
  }
  else
  {
    sub_1698390(&v18, v6, 0);
    if ( v18 != v10 )
    {
LABEL_5:
      sub_169B620(&v18, 1);
      goto LABEL_6;
    }
  }
  sub_169C980(&v18, 1);
LABEL_6:
  v11 = sub_159CCF0(*(_QWORD **)a1, (__int64)v17);
  v12 = v11;
  if ( *(_BYTE *)(a1 + 8) == 16 )
    v12 = sub_15A0390(*(_QWORD *)(a1 + 32), v11);
  if ( v10 != v18 )
  {
    sub_1698460(&v18);
    return v12;
  }
  v14 = v19;
  if ( !v19 )
    return v12;
  v15 = 32LL * *(_QWORD *)(v19 - 8);
  v16 = v19 + v15;
  if ( v19 != v19 + v15 )
  {
    do
    {
      v16 -= 32;
      sub_127D120((_QWORD *)(v16 + 8));
    }
    while ( v14 != v16 );
  }
  j_j_j___libc_free_0_0(v14 - 8);
  return v12;
}
