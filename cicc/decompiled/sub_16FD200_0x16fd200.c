// Function: sub_16FD200
// Address: 0x16fd200
//
_QWORD *__fastcall sub_16FD200(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 *v16; // rax
  _QWORD *v17; // r12
  __int64 v18; // r9
  unsigned __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  _QWORD v27[2]; // [rsp+0h] [rbp-50h] BYREF
  char v28; // [rsp+10h] [rbp-40h]
  char v29; // [rsp+11h] [rbp-3Fh]
  __int64 *v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+28h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a1 + 80);
  if ( v5 )
    return (_QWORD *)v5;
  v7 = sub_16FD110(a1, a2, a3, a4, a5);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  if ( !(unsigned __int8)sub_16F91D0(a1) )
  {
    v11 = (unsigned int *)sub_16FC340(a1, a2, v8, v9, v10);
    v14 = *v11;
    if ( (unsigned int)v14 <= 0x10 )
    {
      v15 = 100609;
      if ( _bittest64(&v15, v14) )
        goto LABEL_8;
      goto LABEL_6;
    }
    if ( (_DWORD)v14 != 17 )
    {
LABEL_6:
      v29 = 1;
      v28 = 3;
      v27[0] = "Unexpected token in Key Value.";
      sub_16F8380(a1, (__int64)v27, (__int64)v11, (__int64)"Unexpected token in Key Value.", v13);
      goto LABEL_8;
    }
    v19 = a1;
    sub_16FC240((__int64)v27, a1, v14, v12, v13);
    if ( v30 != &v31 )
    {
      v19 = v31 + 1;
      j_j___libc_free_0(v30, v31 + 1);
    }
    if ( ((*(_DWORD *)sub_16FC340(a1, v19, v20, v21, v22) - 8) & 0xFFFFFFF7) != 0 )
    {
      v26 = sub_16FD100(a1, v19, v23, v24, v25);
      *(_QWORD *)(a1 + 80) = v26;
      return (_QWORD *)v26;
    }
  }
LABEL_8:
  v16 = (__int64 *)sub_16F82D0(a1);
  v17 = (_QWORD *)sub_145CBF0(v16, 72, 16);
  sub_16FC350((__int64)v17, 0, *(_QWORD *)(a1 + 8), 0, 0, v18, 0);
  *v17 = &unk_49EFDB8;
  *(_QWORD *)(a1 + 80) = v17;
  return v17;
}
