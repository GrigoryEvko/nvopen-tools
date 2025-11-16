// Function: sub_155CEC0
// Address: 0x155cec0
//
__int64 __fastcall sub_155CEC0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-C8h] BYREF
  unsigned __int64 v14[2]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v15[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = *a1;
  v14[0] = (unsigned __int64)v15;
  v14[1] = 0x2000000000LL;
  sub_16BD3E0(v14, a2);
  v5 = v4 + 200;
  if ( !a3 )
  {
    v6 = sub_16BDDE0(v5, v14, &v13);
    if ( v6 )
    {
LABEL_3:
      v7 = v6 - 8;
      goto LABEL_4;
    }
    v12 = sub_22077B0(24);
    v7 = v12;
    if ( v12 )
    {
      *(_QWORD *)(v12 + 8) = 0;
      *(_BYTE *)(v12 + 16) = 0;
      *(_DWORD *)(v12 + 20) = a2;
      *(_QWORD *)v12 = &unk_49ECE28;
      goto LABEL_10;
    }
LABEL_14:
    v10 = v13;
    v7 = 0;
    v11 = 0;
    goto LABEL_11;
  }
  sub_16BD4B0(v14, a3);
  v6 = sub_16BDDE0(v5, v14, &v13);
  if ( v6 )
    goto LABEL_3;
  v9 = sub_22077B0(32);
  v7 = v9;
  if ( !v9 )
    goto LABEL_14;
  *(_QWORD *)(v9 + 8) = 0;
  *(_BYTE *)(v9 + 16) = 1;
  *(_DWORD *)(v9 + 20) = a2;
  *(_QWORD *)(v9 + 24) = a3;
  *(_QWORD *)v9 = &unk_49ECE50;
LABEL_10:
  v10 = v13;
  v11 = v7 + 8;
LABEL_11:
  sub_16BDA20(v5, v11, v10);
LABEL_4:
  if ( (_BYTE *)v14[0] != v15 )
    _libc_free(v14[0]);
  return v7;
}
