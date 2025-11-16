// Function: sub_E770B0
// Address: 0xe770b0
//
__int64 __fastcall sub_E770B0(__int64 *a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // rbx
  char v9; // al
  __int64 v10; // rax
  void (__fastcall *v11)(_QWORD *, char **, __int64 *); // rax
  unsigned int v12; // r15d
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r12
  char *i; // r15
  __int64 v20; // rsi
  unsigned __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v27; // [rsp+30h] [rbp-A0h]
  unsigned int v28; // [rsp+34h] [rbp-9Ch]
  int v29; // [rsp+38h] [rbp-98h]
  char *v30; // [rsp+40h] [rbp-90h] BYREF
  char v31; // [rsp+60h] [rbp-70h]
  char v32; // [rsp+61h] [rbp-6Fh]
  __int64 v33[4]; // [rsp+70h] [rbp-60h] BYREF
  char v34; // [rsp+90h] [rbp-40h]
  char v35; // [rsp+91h] [rbp-3Fh]

  v29 = a3;
  v7 = *a1;
  v8 = a2[1];
  if ( !*a1 )
    v7 = sub_E6C430(v8, (__int64)a2, a3, a4, a5);
  (*(void (__fastcall **)(_QWORD *, __int64))(*a2 + 1296LL))(a2, v7);
  v9 = *(_BYTE *)(v8 + 1906);
  if ( v9 )
  {
    if ( v9 != 1 )
      BUG();
    v27 = 8;
  }
  else
  {
    v27 = 4;
  }
  v10 = *a2;
  v32 = 1;
  v31 = 3;
  v11 = *(void (__fastcall **)(_QWORD *, char **, __int64 *))(v10 + 1288);
  v33[0] = (__int64)"unit length";
  v35 = 1;
  v34 = 3;
  v30 = "debug_line";
  v11(a2, &v30, v33);
  v12 = *(unsigned __int16 *)(v8 + 1904);
  v28 = v12;
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, *(unsigned __int16 *)(v8 + 1904), 2);
  if ( v12 > 4 )
  {
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(
      a2,
      *(unsigned int *)(*(_QWORD *)(v8 + 152) + 8LL),
      1);
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, 0, 1);
  }
  v35 = 1;
  v34 = 3;
  v33[0] = (__int64)"prologue_start";
  v15 = sub_E6C380(v8, v33, 1, v13, v14);
  v35 = 1;
  v33[0] = (__int64)"prologue_end";
  v34 = 3;
  v18 = sub_E6C380(v8, v33, 1, v16, v17);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD))(*a2 + 832LL))(a2, v18, v15, v27);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v15, 0);
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(
    a2,
    *(unsigned int *)(*(_QWORD *)(v8 + 152) + 28LL),
    1);
  if ( v28 > 3 )
    (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, 1, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, 1, 1);
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, SBYTE1(v29), 1);
  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, BYTE2(v29), 1);
  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, a5 + 1, 1);
  for ( i = (char *)a4; (char *)(a4 + a5) != i; ++i )
  {
    v20 = *i;
    (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a2 + 536LL))(a2, v20, 1);
  }
  if ( v28 > 4 )
  {
    sub_E76C90((__int64)a1, a2, a6);
  }
  else
  {
    v21 = *(unsigned int *)(a2[1] + 56LL);
    if ( (unsigned int)v21 <= 0x3A && (v22 = 0x4000C0000200000LL, _bittest64(&v22, v21)) )
      sub_E769A0((__int64)a1, (__int64)a2);
    else
      sub_E76B40((__int64)a1, (__int64)a2);
  }
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v18, 0);
  return v7;
}
