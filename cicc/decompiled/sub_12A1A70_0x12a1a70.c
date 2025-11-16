// Function: sub_12A1A70
// Address: 0x12a1a70
//
__int64 __fastcall sub_12A1A70(__int64 a1, __int64 a2)
{
  int v2; // r14d
  char *v3; // r15
  __int64 v4; // r12
  __int64 v5; // rax
  int v6; // ebx
  int v7; // ecx
  __int64 v8; // r13
  __int64 v9; // r13
  int v11; // [rsp+Ch] [rbp-54h] BYREF
  char *s; // [rsp+10h] [rbp-50h] BYREF
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF

  v2 = sub_12A0C10(a1, *(_QWORD *)(a2 + 160));
  sub_129EFF0(&s, a1, a2);
  v3 = s;
  sub_129E300(*(_DWORD *)(a2 + 64), (char *)&v11);
  v4 = a1 + 16;
  v5 = sub_129F850(a1, *(_DWORD *)(a2 + 64));
  v6 = v11;
  v7 = 0;
  v8 = v5;
  if ( v3 )
    v7 = strlen(v3);
  v9 = sub_15A5B80(v4, v2, (_DWORD)v3, v7, v8, v6, v8);
  sub_15A7340(v4, v9);
  if ( s != (char *)&v13 )
    j_j___libc_free_0(s, v13 + 1);
  return v9;
}
