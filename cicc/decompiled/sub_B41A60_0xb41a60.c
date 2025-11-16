// Function: sub_B41A60
// Address: 0xb41a60
//
__int64 __fastcall sub_B41A60(
        _QWORD **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        char a7,
        int a8,
        char a9)
{
  _QWORD *v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int128 v17; // [rsp-40h] [rbp-90h]
  __int128 v19; // [rsp+10h] [rbp-40h]
  __int128 v20; // [rsp+20h] [rbp-30h]
  __int64 v21; // [rsp+30h] [rbp-20h]

  *(_QWORD *)&v20 = a1;
  v9 = *a1;
  BYTE9(v20) = a7;
  HIDWORD(v20) = a8;
  *(_QWORD *)&v19 = a4;
  LOBYTE(v21) = a9;
  *((_QWORD *)&v19 + 1) = a5;
  BYTE8(v20) = a6;
  v10 = *v9 + 2152LL;
  v11 = sub_BCE3C0(v9, 0);
  *((_QWORD *)&v17 + 1) = a3;
  *(_QWORD *)&v17 = a2;
  return sub_B413D0(v10, v11, v12, v13, v14, v15, v17, v19, v20, v21);
}
