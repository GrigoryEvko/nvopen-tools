// Function: sub_2A758D0
// Address: 0x2a758d0
//
void __fastcall sub_2A758D0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r15
  unsigned __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rdx
  int v12; // eax
  _QWORD *v13; // rdi
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  unsigned __int64 *v16; // r13
  unsigned __int64 *v17; // rdi
  int v18; // ebx
  unsigned __int64 v19[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v3 = *(_QWORD *)(a2 - 64);
  v4 = *(_QWORD *)(a2 - 32);
  v20 = 773;
  v19[0] = (unsigned __int64)sub_BD5D20(a2);
  v19[1] = v5;
  v19[2] = (unsigned __int64)".urem";
  v6 = sub_B504D0(22, v3, v4, (__int64)v19, a2 + 24, 0);
  sub_BD84D0(a2, v6);
  v8 = *(_QWORD *)(a2 + 48);
  v19[0] = v8;
  if ( !v8 )
  {
    v9 = v6 + 48;
    if ( (unsigned __int64 *)(v6 + 48) == v19 )
      goto LABEL_5;
    v14 = *(_QWORD *)(v6 + 48);
    if ( !v14 )
      goto LABEL_5;
LABEL_13:
    sub_B91220(v9, v14);
    goto LABEL_14;
  }
  v9 = v6 + 48;
  sub_B96E90((__int64)v19, v8, 1);
  if ( (unsigned __int64 *)(v6 + 48) == v19 )
  {
    if ( v19[0] )
      sub_B91220((__int64)v19, v19[0]);
    goto LABEL_5;
  }
  v14 = *(_QWORD *)(v6 + 48);
  if ( v14 )
    goto LABEL_13;
LABEL_14:
  v15 = (unsigned __int8 *)v19[0];
  *(_QWORD *)(v6 + 48) = v19[0];
  if ( v15 )
    sub_B976B0((__int64)v19, v15, v9);
LABEL_5:
  v10 = *(_QWORD *)(a1 + 48);
  *(_BYTE *)(a1 + 56) = 1;
  v11 = *(unsigned int *)(v10 + 8);
  v12 = v11;
  if ( *(_DWORD *)(v10 + 12) <= (unsigned int)v11 )
  {
    v16 = (unsigned __int64 *)sub_C8D7D0(v10, v10 + 16, 0, 0x18u, v19, v7);
    v17 = &v16[3 * *(unsigned int *)(v10 + 8)];
    if ( v17 )
    {
      *v17 = 6;
      v17[1] = 0;
      v17[2] = a2;
      if ( a2 != -8192 && a2 != -4096 )
        sub_BD73F0((__int64)v17);
    }
    sub_F17F80(v10, v16);
    v18 = v19[0];
    if ( v10 + 16 != *(_QWORD *)v10 )
      _libc_free(*(_QWORD *)v10);
    ++*(_DWORD *)(v10 + 8);
    *(_QWORD *)v10 = v16;
    *(_DWORD *)(v10 + 12) = v18;
  }
  else
  {
    v13 = (_QWORD *)(*(_QWORD *)v10 + 24 * v11);
    if ( v13 )
    {
      *v13 = 6;
      v13[1] = 0;
      v13[2] = a2;
      if ( a2 != -4096 && a2 != -8192 )
        sub_BD73F0((__int64)v13);
      v12 = *(_DWORD *)(v10 + 8);
    }
    *(_DWORD *)(v10 + 8) = v12 + 1;
  }
}
