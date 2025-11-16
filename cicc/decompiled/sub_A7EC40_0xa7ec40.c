// Function: sub_A7EC40
// Address: 0xa7ec40
//
_BYTE *__fastcall sub_A7EC40(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // r14d
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r9
  _BYTE *v9; // r13
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v13; // r15
  __int64 v15; // rax
  unsigned int *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rsi
  _DWORD v20[4]; // [rsp+10h] [rbp-A0h] BYREF
  char *v21; // [rsp+20h] [rbp-90h] BYREF
  char v22; // [rsp+40h] [rbp-70h]
  char v23; // [rsp+41h] [rbp-6Fh]
  _BYTE v24[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v5 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL);
  v6 = sub_BCB2A0(*(_QWORD *)(a1 + 72));
  v7 = sub_BCDA70(v6, v5 >> 8);
  v25 = 257;
  v9 = (_BYTE *)sub_A7EAA0((unsigned int **)a1, 0x31u, a2, v7, (__int64)v24, 0, (unsigned int)v21, 0);
  if ( a3 <= 4 )
  {
    if ( a3 )
    {
      v10 = 0;
      do
      {
        v20[v10] = v10;
        ++v10;
      }
      while ( a3 != (_DWORD)v10 );
    }
    v11 = *(_QWORD *)(a1 + 80);
    v23 = 1;
    v21 = "extract";
    v22 = 3;
    v12 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v11 + 112LL);
    if ( v12 == sub_9B6630 )
    {
      if ( *v9 > 0x15u )
        goto LABEL_12;
      v13 = sub_AD5CE0(v9, v9, v20, a3, 0, v8);
    }
    else
    {
      v13 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, _QWORD))v12)(v11, v9, v9, v20, a3);
    }
    if ( v13 )
      return (_BYTE *)v13;
LABEL_12:
    v25 = 257;
    v15 = sub_BD2C40(112, unk_3F1FE60);
    v13 = v15;
    if ( v15 )
      sub_B4E9E0(v15, (_DWORD)v9, (_DWORD)v9, (unsigned int)v20, a3, (unsigned int)v24, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v13,
      &v21,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v16 = *(unsigned int **)a1;
    v17 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    while ( (unsigned int *)v17 != v16 )
    {
      v18 = *((_QWORD *)v16 + 1);
      v19 = *v16;
      v16 += 4;
      sub_B99FD0(v13, v19, v18);
    }
    return (_BYTE *)v13;
  }
  return v9;
}
