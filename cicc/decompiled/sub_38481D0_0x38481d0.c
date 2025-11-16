// Function: sub_38481D0
// Address: 0x38481d0
//
__int64 *__fastcall sub_38481D0(__int64 a1, __int64 *a2)
{
  __int64 *v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rax
  __int16 v10; // dx
  __int64 v11; // rax
  __int64 v12; // r9
  _QWORD *v13; // rdi
  unsigned __int8 *v14; // r12
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r13
  bool v18; // al
  __int128 v19; // [rsp-20h] [rbp-90h]
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int128 v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+20h] [rbp-50h] BYREF
  __int64 v23; // [rsp+28h] [rbp-48h]
  __int64 v24; // [rsp+30h] [rbp-40h] BYREF
  __int64 v25; // [rsp+38h] [rbp-38h]

  v4 = (__int64 *)a2[5];
  DWORD2(v21) = 0;
  LODWORD(v23) = 0;
  v5 = v4[5];
  v6 = v4[1];
  *(_QWORD *)&v21 = 0;
  v7 = v4[6];
  v22 = 0;
  v8 = *v4;
  v9 = *(_QWORD *)(v5 + 48) + 16LL * *((unsigned int *)v4 + 12);
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  LOWORD(v24) = v10;
  v25 = v11;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 2) > 7u
      && (unsigned __int16)(v10 - 17) > 0x6Cu
      && (unsigned __int16)(v10 - 176) > 0x1Fu )
    {
      goto LABEL_5;
    }
  }
  else
  {
    v20 = v7;
    v18 = sub_3007070((__int64)&v24);
    v7 = v20;
    if ( !v18 )
    {
LABEL_5:
      sub_375E6F0(a1, v5, v7, (__int64)&v21, (__int64)&v22);
      goto LABEL_6;
    }
  }
  sub_375E510(a1, v5, v7, (__int64)&v21, (__int64)&v22);
LABEL_6:
  v13 = *(_QWORD **)(a1 + 8);
  *((_QWORD *)&v19 + 1) = v6;
  *(_QWORD *)&v19 = v8;
  v24 = 0;
  LODWORD(v25) = 0;
  v14 = sub_3406EB0(v13, 0x170u, (__int64)&v24, 1, 0, v12, v19, v21);
  v16 = v15;
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  sub_33EC010(*(_QWORD **)(a1 + 8), a2, (unsigned __int64)v14, v16, v22, v23);
  return a2;
}
