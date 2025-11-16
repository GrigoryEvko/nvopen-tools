// Function: sub_157EF40
// Address: 0x157ef40
//
__int64 __fastcall sub_157EF40(__int64 a1)
{
  __int64 v2; // r13
  _QWORD *v3; // r14
  _QWORD *v4; // rbx
  unsigned __int64 *v5; // rcx
  unsigned __int64 v6; // rdx
  __int64 v7; // r8
  _QWORD *v8; // r14
  _QWORD *v9; // rbx
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 i; // rbx
  _QWORD *v19; // r13
  __int64 v20; // rax

  if ( *(_WORD *)(a1 + 18) )
  {
    v14 = sub_157E9C0(a1);
    v15 = sub_1643350(v14);
    v16 = sub_159C470(v15, 1, 0);
    v17 = *(_QWORD *)(a1 + 8);
    for ( i = v16; v17; v17 = *(_QWORD *)(a1 + 8) )
    {
      v19 = (_QWORD *)sub_1648700(v17);
      v20 = sub_15A3BA0(i, *v19, 0);
      sub_164D160(v19, v20);
      sub_159D850(v19);
    }
  }
  v2 = a1 + 40;
  sub_157EE90(a1);
  v3 = *(_QWORD **)(a1 + 48);
  if ( (_QWORD *)(a1 + 40) != v3 )
  {
    do
    {
      v4 = v3;
      v3 = (_QWORD *)v3[1];
      sub_157EA20(a1 + 40, (__int64)(v4 - 3));
      v5 = (unsigned __int64 *)v4[1];
      v6 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
      *v5 = v6 | *v5 & 7;
      *(_QWORD *)(v6 + 8) = v5;
      *v4 &= 7uLL;
      v4[1] = 0;
      sub_164BEC0(v4 - 3, v4 - 3, v6, v5, v7);
    }
    while ( (_QWORD *)v2 != v3 );
    v8 = *(_QWORD **)(a1 + 48);
    while ( (_QWORD *)v2 != v8 )
    {
      v9 = v8;
      v8 = (_QWORD *)v8[1];
      sub_157EA20(a1 + 40, (__int64)(v9 - 3));
      v10 = (unsigned __int64 *)v9[1];
      v11 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
      *v10 = v11 | *v10 & 7;
      *(_QWORD *)(v11 + 8) = v10;
      *v9 &= 7uLL;
      v9[1] = 0;
      sub_164BEC0(v9 - 3, v9 - 3, v11, v10, v12);
    }
  }
  return sub_164BE60(a1);
}
