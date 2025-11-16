// Function: sub_926480
// Address: 0x926480
//
__int64 __fastcall sub_926480(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, char a5)
{
  int v6; // edx
  __int64 v7; // rax
  unsigned __int8 v8; // al
  int v9; // r9d
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned int *v12; // rbx
  unsigned int *v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rsi
  bool v17; // al
  int v18; // [rsp+8h] [rbp-98h]
  int v19; // [rsp+Ch] [rbp-94h]
  int v20; // [rsp+Ch] [rbp-94h]
  int v21; // [rsp+Ch] [rbp-94h]
  char *v22; // [rsp+10h] [rbp-90h] BYREF
  char v23; // [rsp+30h] [rbp-70h]
  char v24; // [rsp+31h] [rbp-6Fh]
  char v25; // [rsp+40h] [rbp-60h] BYREF
  __int16 v26; // [rsp+60h] [rbp-40h]

  v6 = 1;
  v24 = 1;
  v22 = "tmp";
  v23 = 3;
  if ( !a5 )
  {
    v6 = unk_4D0463C;
    if ( unk_4D0463C )
    {
      v21 = a4;
      v17 = sub_90AA40(*(_QWORD *)(a1 + 32), a2);
      LODWORD(a4) = v21;
      v6 = v17;
    }
  }
  if ( (_DWORD)a4 )
  {
    _BitScanReverse64((unsigned __int64 *)&a4, (unsigned int)a4);
    v9 = (unsigned __int8)(63 - (a4 ^ 0x3F));
  }
  else
  {
    v19 = v6;
    v7 = sub_AA4E30(*(_QWORD *)(a1 + 96));
    v8 = sub_AE5020(v7, a3);
    v6 = v19;
    v9 = v8;
  }
  v18 = v9;
  v20 = v6;
  v26 = 257;
  v10 = sub_BD2C40(80, unk_3F10A14);
  v11 = v10;
  if ( v10 )
    sub_B4D190(v10, a3, a2, (unsigned int)&v25, v20, v18, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
    *(_QWORD *)(a1 + 136),
    v11,
    &v22,
    *(_QWORD *)(a1 + 104),
    *(_QWORD *)(a1 + 112));
  v12 = *(unsigned int **)(a1 + 48);
  v13 = &v12[4 * *(unsigned int *)(a1 + 56)];
  while ( v13 != v12 )
  {
    v14 = *((_QWORD *)v12 + 1);
    v15 = *v12;
    v12 += 4;
    sub_B99FD0(v11, v15, v14);
  }
  return v11;
}
