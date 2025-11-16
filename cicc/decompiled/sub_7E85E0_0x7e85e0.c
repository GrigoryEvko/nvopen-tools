// Function: sub_7E85E0
// Address: 0x7e85e0
//
_BYTE *__fastcall sub_7E85E0(__m128i *a1, __int64 *a2, int a3)
{
  __int64 i; // rbx
  __int64 *v7; // r13
  __int64 v8; // rsi
  void *v9; // rbx
  _BYTE *v10; // rax
  __int64 *v11; // rax
  void *v12; // r12
  __int64 v13; // rax
  _BYTE *v14; // rdi
  _QWORD *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r15
  void *v18; // rbx
  _BYTE *v19; // rax
  __int64 *v20; // rax
  void *v21; // r13
  __int64 v22; // rax
  __int64 *v23; // rax
  _DWORD v24[13]; // [rsp+Ch] [rbp-34h] BYREF

  for ( i = a1->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  sub_7E3EE0(i);
  if ( a3 )
    return sub_7E2400(a1, *(_QWORD *)(i + 160), a2[13], a2[5]);
  v7 = sub_7E7F80(a1, 0, v24, 0);
  if ( v24[0] )
  {
    v15 = sub_7E8090(a1, 0);
    v16 = a2[18];
    v17 = (__int64)v15;
    v18 = sub_7E4640(v7, v16);
    v19 = sub_73E1B0(v17, v16);
    v20 = (__int64 *)sub_7E23D0(v19);
    v20[2] = (__int64)v18;
    v21 = sub_73DBF0(0x32u, *v20, (__int64)v20);
    v22 = sub_72D2E0((_QWORD *)a2[5]);
    v23 = (__int64 *)sub_73E110((__int64)v21, v22);
    v14 = sub_73DF90((__int64)a1, v23);
  }
  else
  {
    v8 = a2[18];
    v9 = sub_7E4640(v7, v8);
    v10 = sub_73E1B0((__int64)a1, v8);
    v11 = (__int64 *)sub_7E23D0(v10);
    v11[2] = (__int64)v9;
    v12 = sub_73DBF0(0x32u, *v11, (__int64)v11);
    v13 = sub_72D2E0((_QWORD *)a2[5]);
    v14 = sub_73E110((__int64)v12, v13);
  }
  return sub_73DCD0(v14);
}
