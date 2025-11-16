// Function: sub_10E7FB0
// Address: 0x10e7fb0
//
__int64 __fastcall sub_10E7FB0(unsigned int **a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  char v9; // al
  __int64 v10; // rsi
  __int64 *v11; // rdi
  char v12; // dl
  __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // rax
  char v16; // dl
  __int64 *v17; // rcx
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-60h] BYREF
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 *v25; // [rsp+18h] [rbp-48h]
  __int16 v26; // [rsp+30h] [rbp-30h]

  v25 = &v22;
  v3 = *(_QWORD *)(a2 + 16);
  v24 = 0;
  if ( !v3 || *(_QWORD *)(v3 + 8) || *(_BYTE *)a2 != 59 )
    return 0;
  v9 = sub_995B10(&v24, *(_QWORD *)(a2 - 64));
  v10 = *(_QWORD *)(a2 - 32);
  if ( v9 && v10 )
  {
    *v25 = v10;
    goto LABEL_9;
  }
  if ( !(unsigned __int8)sub_995B10(&v24, v10) )
    return 0;
  v21 = *(_QWORD *)(a2 - 64);
  if ( !v21 )
    return 0;
  *v25 = v21;
LABEL_9:
  v11 = (__int64 *)*a1;
  v12 = 0;
  v13 = *(_QWORD *)(v22 + 16);
  if ( v13 )
    v12 = *(_QWORD *)(v13 + 8) == 0;
  LOBYTE(v23) = 0;
  if ( sub_F13D80(v11, v22, v12, 0, &v23, 0) )
    return 0;
  v14 = (__int64 *)*a1;
  v15 = *(_QWORD *)(a3 + 16);
  v16 = 0;
  v17 = (__int64 *)*((_QWORD *)*a1 + 4);
  if ( v15 )
    v16 = *(_QWORD *)(v15 + 8) == 0;
  LOBYTE(v24) = 0;
  v4 = sub_F13D80(v14, a3, v16, v17, &v24, 0);
  if ( !v4 )
    return v4;
  v18 = sub_9905C0(*a1[1]);
  v19 = *((_QWORD *)*a1 + 4);
  HIDWORD(v23) = 0;
  v26 = 257;
  v20 = sub_B33C40(v19, v18, v22, v4, (unsigned int)v23, (__int64)&v24);
  v26 = 257;
  return sub_B50640(v20, (__int64)&v24, 0, 0);
}
