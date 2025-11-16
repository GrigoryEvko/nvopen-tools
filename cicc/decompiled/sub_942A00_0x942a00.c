// Function: sub_942A00
// Address: 0x942a00
//
__int64 __fastcall sub_942A00(__int64 a1, __int64 a2)
{
  int v2; // r14d
  char *v3; // r15
  __int64 v4; // r12
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  int v8; // r9d
  __int64 v9; // rax
  int v10; // ebx
  int v11; // ecx
  __int64 v12; // r13
  __int64 v13; // r13
  int v15; // [rsp+Ch] [rbp-54h] BYREF
  char *s; // [rsp+10h] [rbp-50h] BYREF
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF

  v2 = sub_941B90(a1, *(_QWORD *)(a2 + 160));
  sub_93FAB0(&s, a1, a2);
  v3 = s;
  sub_93ED80(*(_DWORD *)(a2 + 64), (char *)&v15);
  v4 = a1 + 16;
  v9 = sub_9405D0(a1, *(_DWORD *)(a2 + 64), v5, v6, v7, v8);
  v10 = v15;
  v11 = 0;
  v12 = v9;
  if ( v3 )
    v11 = strlen(v3);
  v13 = sub_ADCB10(v4, v2, (_DWORD)v3, v11, v12, v10, v12, 0, 0, 0);
  sub_ADDCE0(v4, v13);
  if ( s != (char *)&v17 )
    j_j___libc_free_0(s, v17 + 1);
  return v13;
}
