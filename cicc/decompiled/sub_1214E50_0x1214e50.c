// Function: sub_1214E50
// Address: 0x1214e50
//
void __fastcall sub_1214E50(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12

  v3 = sub_1214C20(a1, a2);
  v5 = v4;
  v6 = v3;
  if ( v3 == *(_QWORD *)(a1 + 24) && v4 == a1 + 8 )
  {
    sub_1207A20(*(_QWORD **)(a1 + 16));
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = v5;
    *(_QWORD *)(a1 + 32) = v5;
    *(_QWORD *)(a1 + 40) = 0;
  }
  else if ( v4 != v3 )
  {
    do
    {
      v7 = v6;
      v6 = sub_220EF30(v6);
      v8 = sub_220F330(v7, a1 + 8);
      v9 = *(_QWORD *)(v8 + 32);
      v10 = v8;
      if ( v9 != v8 + 48 )
        j_j___libc_free_0(v9, *(_QWORD *)(v8 + 48) + 1LL);
      j_j___libc_free_0(v10, 80);
      --*(_QWORD *)(a1 + 40);
    }
    while ( v5 != v6 );
  }
}
