// Function: sub_2477F90
// Address: 0x2477f90
//
void __fastcall sub_2477F90(__int64 *a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // r9
  __int64 v5; // rsi
  __int64 v6; // r15
  _BYTE *v7; // rax
  _BYTE *v8; // r15
  __int64 v9; // rcx
  __int64 v10; // r8
  bool v11; // al
  __int64 v12; // r9
  __int64 **v13; // rax
  unsigned __int64 v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-108h]
  int v19; // [rsp+18h] [rbp-F8h]
  _QWORD v20[4]; // [rsp+20h] [rbp-F0h] BYREF
  char v21; // [rsp+40h] [rbp-D0h]
  char v22; // [rsp+41h] [rbp-CFh]
  unsigned int *v23[2]; // [rsp+50h] [rbp-C0h] BYREF
  char v24; // [rsp+60h] [rbp-B0h] BYREF
  void *v25; // [rsp+D0h] [rbp-40h]

  sub_23D0AB0((__int64)v23, a2, 0, 0, 0);
  v3 = *(_DWORD *)(a2 + 4);
  v22 = 1;
  v21 = 3;
  v4 = *(_QWORD *)(a2 - 32LL * (v3 & 0x7FFFFFF));
  v20[0] = "_mscz_bs";
  v5 = v4;
  v18 = v4;
  v6 = sub_246F3F0((__int64)a1, v4);
  v7 = (_BYTE *)sub_AD6530(*(_QWORD *)(v6 + 8), v5);
  v8 = (_BYTE *)sub_92B530(v23, 0x21u, v6, v7, (__int64)v20);
  v11 = sub_AD7890(
          *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
          33,
          *(_DWORD *)(a2 + 4) & 0x7FFFFFF,
          v9,
          v10);
  v12 = v18;
  if ( !v11 )
  {
    v22 = 1;
    v20[0] = "_mscz_bzp";
    v21 = 3;
    v15 = (_BYTE *)sub_AD6530(*(_QWORD *)(v18 + 8), 33);
    v16 = (_BYTE *)sub_92B530(v23, 0x20u, v18, v15, (__int64)v20);
    v22 = 1;
    v20[0] = "_mscz_bs";
    v21 = 3;
    v17 = sub_A82480(v23, v8, v16, (__int64)v20);
    v12 = v18;
    v8 = (_BYTE *)v17;
  }
  v22 = 1;
  v20[0] = "_mscz_os";
  v21 = 3;
  v13 = (__int64 **)sub_2463540(a1, *(_QWORD *)(v12 + 8));
  v14 = sub_24633A0((__int64 *)v23, 0x28u, (unsigned __int64)v8, v13, (__int64)v20, 0, v19, 0);
  sub_246EF60((__int64)a1, a2, v14);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v25 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v23[0] != &v24 )
    _libc_free((unsigned __int64)v23[0]);
}
