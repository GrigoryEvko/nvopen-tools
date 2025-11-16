// Function: sub_DCFC90
// Address: 0xdcfc90
//
__int64 __fastcall sub_DCFC90(__int64 a1, __int64 **a2, __int64 *a3)
{
  __int64 *v4; // r13
  __int64 *v5; // rdi
  __int64 *v6; // rsi
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rdx
  bool v12; // zf
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-38h] BYREF
  __int64 v16; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v17[5]; // [rsp+18h] [rbp-28h] BYREF

  v4 = *a2;
  v5 = *(__int64 **)a1;
  v6 = *a2;
  v15 = *a3;
  v7 = sub_DCFA50(v5, (__int64)v6, v15);
  LOBYTE(v8) = sub_D968A0((__int64)v7);
  v9 = v8;
  if ( (_BYTE)v8 || (unsigned __int16)(*((_WORD *)v4 + 12) - 9) > 3u )
    return v9;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)v4[4];
  v12 = *(_QWORD *)(v10 + 16) == 0;
  v16 = v11;
  if ( v12 )
    goto LABEL_8;
  v6 = &v16;
  v7 = (_QWORD *)v10;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64 *, __int64 *))(v10 + 24))(v10, &v16, &v15) )
    return v9;
  v14 = *(_QWORD *)(a1 + 8);
  v11 = *(_QWORD *)(v4[4] + 8);
  v12 = *(_QWORD *)(v14 + 16) == 0;
  v17[0] = v11;
  if ( v12 )
LABEL_8:
    sub_4263D6(v7, v6, v11);
  return (*(unsigned int (__fastcall **)(__int64, _QWORD *, __int64 *))(v14 + 24))(v14, v17, &v15);
}
