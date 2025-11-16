// Function: sub_2C8F8E0
// Address: 0x2c8f8e0
//
__int64 __fastcall sub_2C8F8E0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 v9; // rax
  __int64 *v10; // r14
  __int64 v11; // r9
  _DWORD *v12; // r8
  __int64 v13; // rcx
  _DWORD *v14; // rdx
  int v15; // esi
  __int64 v16; // r15
  __int64 v17; // rbx
  int v18; // r12d
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v23; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+8h] [rbp-58h] BYREF
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  unsigned int v27; // [rsp+20h] [rbp-40h]

  v5 = *a2;
  if ( !*a2 )
    goto LABEL_4;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
  if ( v6 == sub_23CE270 )
    BUG();
  v7 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a3);
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 144LL);
  if ( v8 != sub_2C8F680 )
    v9 = ((__int64 (__fastcall *)(__int64))v8)(v7);
  else
LABEL_4:
    v9 = 0;
  v10 = &v24;
  v23 = v9;
  v24 = 1;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  sub_A09770((__int64)&v24, 0);
  if ( !v27 )
  {
    LODWORD(v26) = v26 + 1;
    BUG();
  }
  v11 = 1;
  v12 = 0;
  v13 = ((_WORD)v27 - 1) & 0x940;
  v14 = (_DWORD *)(v25 + 8 * v13);
  v15 = *v14;
  if ( *v14 != 64 )
  {
    while ( v15 != -1 )
    {
      if ( !v12 && v15 == -2 )
        v12 = v14;
      v13 = (v27 - 1) & ((_DWORD)v11 + (_DWORD)v13);
      v14 = (_DWORD *)(v25 + 8LL * (unsigned int)v13);
      v15 = *v14;
      if ( *v14 == 64 )
        goto LABEL_7;
      v11 = (unsigned int)(v11 + 1);
    }
    if ( v12 )
      v14 = v12;
  }
LABEL_7:
  LODWORD(v26) = v26 + 1;
  if ( *v14 != -1 )
    --HIDWORD(v26);
  *(_QWORD *)v14 = 0x2000000040LL;
  if ( v23 && *(_DWORD *)(v23 + 40) )
    v10 = (__int64 *)(v23 + 24);
  v16 = a3 + 72;
  v17 = *(_QWORD *)(a3 + 80);
  v18 = 0;
  if ( v16 == v17 )
  {
    v20 = a1 + 32;
    v21 = a1 + 80;
LABEL_19:
    *(_QWORD *)(a1 + 8) = v20;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v21;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_16;
  }
  do
  {
    v19 = v17;
    v17 = *(_QWORD *)(v17 + 8);
    v18 |= sub_3193470(v19 - 24, v10, 0, v13, v12, v11);
  }
  while ( v16 != v17 );
  v20 = a1 + 32;
  v21 = a1 + 80;
  if ( !(_BYTE)v18 )
    goto LABEL_19;
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v20;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v21;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
LABEL_16:
  sub_C7D6A0(v25, 8LL * v27, 4);
  return a1;
}
