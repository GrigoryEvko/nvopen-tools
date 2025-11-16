// Function: sub_3219F60
// Address: 0x3219f60
//
void __fastcall sub_3219F60(char **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  unsigned __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  char *v15; // rdi
  _BYTE v16[8]; // [rsp+10h] [rbp-C0h] BYREF
  unsigned __int64 v17; // [rsp+18h] [rbp-B8h]
  _BYTE v18[8]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v19; // [rsp+38h] [rbp-98h]
  char *v20; // [rsp+50h] [rbp-80h]
  char *v21[2]; // [rsp+58h] [rbp-78h] BYREF
  _BYTE v22[48]; // [rsp+68h] [rbp-68h] BYREF
  char v23; // [rsp+98h] [rbp-38h]

  v6 = (__int64)(a1 + 1);
  v21[0] = v22;
  v7 = *a1;
  v8 = *((unsigned int *)a1 + 4);
  v21[1] = (char *)0x200000000LL;
  v20 = v7;
  if ( (_DWORD)v8 )
  {
    sub_3218940((__int64)v21, a1 + 1, v8, (__int64)v21, a5, a6);
    v7 = v20;
  }
  v23 = *((_BYTE *)a1 + 72);
  while ( 1 )
  {
    v9 = v6 - 8;
    sub_AF47B0((__int64)v16, *((unsigned __int64 **)v7 + 2), *((unsigned __int64 **)v7 + 3));
    v10 = v17;
    sub_AF47B0(
      (__int64)v18,
      *(unsigned __int64 **)(*(_QWORD *)(v6 - 88) + 16LL),
      *(unsigned __int64 **)(*(_QWORD *)(v6 - 88) + 24LL));
    if ( v10 >= v19 )
      break;
    *(_QWORD *)(v6 - 8) = *(_QWORD *)(v6 - 88);
    sub_3218940(v6, (char **)(v6 - 80), v11, v12, v13, v14);
    *(_BYTE *)(v6 + 64) = *(_BYTE *)(v6 - 16);
    v7 = v20;
    v6 -= 80;
  }
  *(_QWORD *)v9 = v20;
  sub_3218940(v6, v21, v11, v12, v13, v14);
  v15 = v21[0];
  *(_BYTE *)(v9 + 72) = v23;
  if ( v15 != v22 )
    _libc_free((unsigned __int64)v15);
}
