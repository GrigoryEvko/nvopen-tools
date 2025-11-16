// Function: sub_2E32300
// Address: 0x2e32300
//
__int64 __fastcall sub_2E32300(__int64 *a1, char a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // r15
  __int64 (*v8)(); // rax
  __int64 *v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  int v12; // edx
  __int64 (*v13)(); // rdx
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v18; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v19; // [rsp+18h] [rbp-E8h] BYREF
  _BYTE *v20; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v21; // [rsp+28h] [rbp-D8h]
  _BYTE v22[208]; // [rsp+30h] [rbp-D0h] BYREF

  v2 = a1[1];
  if ( v2 == a1[4] + 320 || !sub_2E322C0((__int64)a1, a1[1]) )
    return 0;
  v18 = 0;
  v21 = 0x400000000LL;
  v4 = a1[4];
  v19 = 0;
  v5 = *(_QWORD *)(v4 + 16);
  v20 = v22;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 128LL);
  if ( v6 == sub_2DAC790 )
    BUG();
  v7 = v6();
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 344LL);
  if ( v8 != sub_2DB1AE0
    && !((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64 *, __int64 *, _BYTE **, _QWORD))v8)(
          v7,
          a1,
          &v18,
          &v19,
          &v20,
          0) )
  {
    if ( !v18 || a2 && (v18 == v2 || v19 == v2) || (_DWORD)v21 && !v19 )
      goto LABEL_22;
    goto LABEL_36;
  }
  v9 = (__int64 *)(a1[6] & 0xFFFFFFFFFFFFFFF8LL);
  v10 = (unsigned __int64)v9;
  if ( v9 == a1 + 6 )
    goto LABEL_22;
  if ( !v9 )
    BUG();
  v11 = *v9;
  v12 = *(_DWORD *)(v10 + 44);
  if ( (v11 & 4) != 0 )
  {
    if ( (v12 & 4) != 0 )
      goto LABEL_27;
  }
  else if ( (v12 & 4) != 0 )
  {
    while ( 1 )
    {
      v10 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      LOBYTE(v12) = *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 44);
      if ( (v12 & 4) == 0 )
        break;
      v11 = *(_QWORD *)v10;
    }
  }
  if ( (v12 & 8) == 0 )
  {
LABEL_27:
    if ( (*(_BYTE *)(*(_QWORD *)(v10 + 16) + 25LL) & 1) == 0 )
      goto LABEL_22;
    v14 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
    v13 = *(__int64 (**)())(*(_QWORD *)v7 + 920LL);
    goto LABEL_15;
  }
  if ( !(unsigned __int8)sub_2E88A90(v10, 256, 1) )
    goto LABEL_22;
  v13 = *(__int64 (**)())(*(_QWORD *)v7 + 920LL);
  v14 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v14 )
    BUG();
LABEL_15:
  v15 = *(_QWORD *)v14;
  if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      v14 = v16;
      if ( (*(_BYTE *)(v16 + 44) & 4) == 0 )
        break;
      v15 = *(_QWORD *)v16;
    }
  }
  if ( v13 != sub_2DB1B30 && ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v13)(v7, v14) )
    goto LABEL_22;
LABEL_36:
  v2 = 0;
LABEL_22:
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v2;
}
