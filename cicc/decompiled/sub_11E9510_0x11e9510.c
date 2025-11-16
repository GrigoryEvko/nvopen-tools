// Function: sub_11E9510
// Address: 0x11e9510
//
__int64 __fastcall sub_11E9510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // rax
  _BYTE *v9; // rsi
  _QWORD *v10; // rdi
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r9
  unsigned __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 **v23; // r14
  unsigned int v24; // eax
  unsigned __int64 v25; // rax
  __int64 v27; // [rsp+8h] [rbp-A8h]
  unsigned int v28; // [rsp+18h] [rbp-98h]
  __int64 v29; // [rsp+18h] [rbp-98h]
  unsigned int v30; // [rsp+18h] [rbp-98h]
  char *v31; // [rsp+20h] [rbp-90h] BYREF
  char v32; // [rsp+40h] [rbp-70h]
  char v33; // [rsp+41h] [rbp-6Fh]
  _QWORD v34[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v35; // [rsp+70h] [rbp-40h]

  sub_11E6850(a1, (unsigned __int8 *)a2, a3, 3);
  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v7 = *(_QWORD *)(a2 + 32 * (1 - v6));
  v8 = (_QWORD *)(a2 + 32 * (2 - v6));
  if ( *(_BYTE *)v7 != 17 )
    return 0;
  v9 = (_BYTE *)*v8;
  if ( *(_BYTE *)*v8 != 17 )
    return 0;
  v10 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  v11 = (_QWORD *)*((_QWORD *)v9 + 3);
  if ( *((_DWORD *)v9 + 8) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = (_QWORD)v10 * (_QWORD)v11;
  if ( v12 )
  {
    if ( v12 == 1 && !*(_QWORD *)(a2 + 16) )
    {
      v27 = *(_QWORD *)(a2 - 32 * v6);
      v13 = sub_BCB2B0(*(_QWORD **)(a3 + 72));
      v14 = *(_QWORD *)(a3 + 48);
      v33 = 1;
      v15 = v13;
      v32 = 3;
      v31 = "char";
      v16 = sub_AA4E30(v14);
      v35 = 257;
      v28 = (unsigned __int8)sub_AE5020(v16, v15);
      v17 = sub_BD2C40(80, unk_3F10A14);
      v18 = v28;
      v19 = (unsigned __int64)v17;
      if ( v17 )
        sub_B4D190((__int64)v17, v15, v27, (__int64)v34, 0, v28, 0, 0);
      (*(void (__fastcall **)(_QWORD, unsigned __int64, char **, _QWORD, _QWORD, __int64))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v19,
        &v31,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64),
        v18);
      v20 = *(_QWORD *)a3;
      v29 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v29 )
      {
        do
        {
          v21 = *(_QWORD *)(v20 + 8);
          v22 = *(_DWORD *)v20;
          v20 += 16;
          sub_B99FD0(v19, v22, v21);
        }
        while ( v29 != v20 );
      }
      v23 = (__int64 **)sub_BCD140(*(_QWORD **)(a3 + 72), *(_DWORD *)(**(_QWORD **)(a1 + 24) + 172LL));
      v35 = 259;
      v34[0] = "chari";
      v30 = sub_BCB060(*(_QWORD *)(v19 + 8));
      v24 = sub_BCB060((__int64)v23);
      v25 = sub_11DB4B0((__int64 *)a3, v24 < v30 ? 38 : 40, v19, v23, (__int64)v34, 0, (int)v31, 0);
      if ( sub_11CD1C0(
             v25,
             *(_QWORD *)(a2 + 32 * (3LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))),
             a3,
             *(__int64 **)(a1 + 24)) )
      {
        return sub_AD64C0(*(_QWORD *)(a2 + 8), 1, 0);
      }
    }
    return 0;
  }
  return sub_AD64C0(*(_QWORD *)(a2 + 8), 0, 0);
}
