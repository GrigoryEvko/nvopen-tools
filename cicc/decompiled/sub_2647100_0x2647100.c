// Function: sub_2647100
// Address: 0x2647100
//
void __fastcall sub_2647100(__int64 a1, __int64 *a2, __int64 a3, int a4)
{
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int8 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 *v17; // [rsp+8h] [rbp-2D8h]
  unsigned __int64 v18[4]; // [rsp+10h] [rbp-2D0h] BYREF
  unsigned __int64 v19[6]; // [rsp+30h] [rbp-2B0h] BYREF
  unsigned __int64 v20[4]; // [rsp+60h] [rbp-280h] BYREF
  unsigned __int64 v21[6]; // [rsp+80h] [rbp-260h] BYREF
  unsigned __int64 v22[4]; // [rsp+B0h] [rbp-230h] BYREF
  unsigned __int64 v23[6]; // [rsp+D0h] [rbp-210h] BYREF
  _QWORD v24[10]; // [rsp+100h] [rbp-1E0h] BYREF
  unsigned __int64 *v25; // [rsp+150h] [rbp-190h]
  unsigned int v26; // [rsp+158h] [rbp-188h]
  char v27; // [rsp+160h] [rbp-180h] BYREF

  if ( a4 )
  {
    v13 = *a2;
    v14 = *(_QWORD *)(*a2 - 32) == 0;
    *(_QWORD *)(*a2 + 80) = *(_QWORD *)(a3 + 24);
    if ( !v14 )
    {
      v15 = *(_QWORD *)(v13 - 24);
      **(_QWORD **)(v13 - 16) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v13 - 16);
    }
    *(_QWORD *)(v13 - 32) = a3;
    v16 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v13 - 24) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = v13 - 24;
    *(_QWORD *)(v13 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = v13 - 32;
  }
  v5 = sub_B43CB0(*a2);
  v17 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 360))(*(_QWORD *)(a1 + 368), v5);
  sub_B174A0((__int64)v24, (__int64)"memprof-context-disambiguation", (__int64)"MemprofCall", 11, *a2);
  sub_B16080((__int64)v22, "Call", 4, (unsigned __int8 *)*a2);
  v6 = sub_2647050((__int64)v24, (__int64)v22);
  sub_B18290(v6, " in clone ", 0xAu);
  v7 = (unsigned __int8 *)sub_B43CB0(*a2);
  sub_B16080((__int64)v20, "Caller", 6, v7);
  v8 = sub_23FD640(v6, (__int64)v20);
  sub_B18290(v8, " assigned to call function clone ", 0x21u);
  sub_B16080((__int64)v18, "Callee", 6, (unsigned __int8 *)a3);
  v9 = sub_23FD640(v8, (__int64)v18);
  sub_1049740(v17, v9);
  sub_2240A30(v19);
  sub_2240A30(v18);
  sub_2240A30(v21);
  sub_2240A30(v20);
  sub_2240A30(v23);
  sub_2240A30(v22);
  v10 = v25;
  v24[0] = &unk_49D9D40;
  v11 = &v25[10 * v26];
  if ( v25 != v11 )
  {
    do
    {
      v11 -= 10;
      v12 = v11[4];
      if ( (unsigned __int64 *)v12 != v11 + 6 )
        j_j___libc_free_0(v12);
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        j_j___libc_free_0(*v11);
    }
    while ( v10 != v11 );
    v11 = v25;
  }
  if ( v11 != (unsigned __int64 *)&v27 )
    _libc_free((unsigned __int64)v11);
}
