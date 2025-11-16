// Function: sub_A80050
// Address: 0xa80050
//
__int64 __fastcall sub_A80050(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v3; // r15
  unsigned int v5; // ebx
  _BYTE *v7; // rax
  __int64 v8; // rdi
  _BYTE *v9; // r12
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, _BYTE *); // rax
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // r12d
  _DWORD *v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // r9
  __int64 v18; // rdi
  _BYTE *v19; // rbx
  __int64 (__fastcall *v20)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v21; // r14
  __int64 v22; // rax
  unsigned int *v24; // r12
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  unsigned int *v29; // r15
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rsi
  _DWORD v33[8]; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v34[8]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v35; // [rsp+50h] [rbp-70h]
  _BYTE v36[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v37; // [rsp+80h] [rbp-40h]

  v3 = a2;
  v5 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL);
  if ( a3 && (*a3 > 0x15u || !(unsigned __int8)sub_AD7930(a3)) )
  {
    v35 = 257;
    v7 = sub_A7EC40(a1, (__int64)a3, v5);
    v8 = *(_QWORD *)(a1 + 80);
    v9 = v7;
    v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, _BYTE *))(*(_QWORD *)v8 + 16LL);
    if ( v10 == sub_9202E0 )
    {
      if ( *(_BYTE *)a2 > 0x15u || *v9 > 0x15u )
        goto LABEL_25;
      if ( (unsigned __int8)sub_AC47B0(28) )
        v11 = sub_AD5570(28, a2, v9, 0, 0);
      else
        v11 = sub_AABE40(28, a2, v9);
    }
    else
    {
      v11 = v10(v8, 28u, (_BYTE *)a2, v9);
    }
    if ( v11 )
    {
LABEL_10:
      v3 = v11;
      goto LABEL_11;
    }
LABEL_25:
    v37 = 257;
    v11 = sub_B504D0(28, a2, v9, v36, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v11,
      v34,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v24 = *(unsigned int **)a1;
    v25 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v25 )
    {
      do
      {
        v26 = *((_QWORD *)v24 + 1);
        v27 = *v24;
        v24 += 4;
        sub_B99FD0(v11, v27, v26);
      }
      while ( (unsigned int *)v25 != v24 );
    }
    goto LABEL_10;
  }
LABEL_11:
  if ( v5 > 7 )
  {
    v21 = v3;
    v13 = v5;
    v37 = 257;
    goto LABEL_22;
  }
  v12 = 0;
  if ( v5 )
  {
    do
    {
      v33[v12] = v12;
      ++v12;
    }
    while ( v5 != (_DWORD)v12 );
  }
  v13 = v5;
  v14 = &v33[v5];
  do
  {
    v15 = v13++;
    *v14++ = v5 + v15 % v5;
  }
  while ( v13 != 8 );
  v35 = 257;
  v16 = sub_AD6530(*(_QWORD *)(v3 + 8));
  v18 = *(_QWORD *)(a1 + 80);
  v19 = (_BYTE *)v16;
  v20 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v18 + 112LL);
  if ( v20 == sub_9B6630 )
  {
    if ( *(_BYTE *)v3 > 0x15u || *v19 > 0x15u )
      goto LABEL_28;
    v21 = sub_AD5CE0(v3, v19, v33, 8, 0, v17);
  }
  else
  {
    v21 = ((__int64 (__fastcall *)(__int64, __int64, _BYTE *, _DWORD *, __int64))v20)(v18, v3, v19, v33, 8);
  }
  if ( !v21 )
  {
LABEL_28:
    v37 = 257;
    v28 = sub_BD2C40(112, unk_3F1FE60);
    v21 = v28;
    if ( v28 )
      sub_B4E9E0(v28, v3, (_DWORD)v19, (unsigned int)v33, 8, (unsigned int)v36, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, unsigned int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
      *(_QWORD *)(a1 + 88),
      v21,
      v34,
      *(_QWORD *)(a1 + 56),
      *(_QWORD *)(a1 + 64));
    v29 = *(unsigned int **)a1;
    v30 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v30 )
    {
      do
      {
        v31 = *((_QWORD *)v29 + 1);
        v32 = *v29;
        v29 += 4;
        sub_B99FD0(v21, v32, v31);
      }
      while ( (unsigned int *)v30 != v29 );
    }
  }
  v37 = 257;
LABEL_22:
  v22 = sub_BCD140(*(_QWORD *)(a1 + 72), v13);
  return sub_A7EAA0((unsigned int **)a1, 0x31u, v21, v22, (__int64)v36, 0, v34[0], 0);
}
