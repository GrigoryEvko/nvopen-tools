// Function: sub_2C1ECD0
// Address: 0x2c1ecd0
//
__int64 *__fastcall sub_2C1ECD0(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 *v5; // rdi
  char v6; // al
  __int64 v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rax
  _BYTE *v15; // r10
  __int64 v16; // rax
  __int64 v17; // r10
  _QWORD *v19; // rax
  _QWORD *v20; // r10
  __int64 v21; // [rsp+0h] [rbp-B0h]
  __int64 v22; // [rsp+0h] [rbp-B0h]
  char v23; // [rsp+Ch] [rbp-A4h]
  __int64 **v24; // [rsp+10h] [rbp-A0h]
  char v25; // [rsp+18h] [rbp-98h]
  __int64 v26; // [rsp+18h] [rbp-98h]
  __int64 v27; // [rsp+18h] [rbp-98h]
  _QWORD *v28; // [rsp+18h] [rbp-98h]
  __int64 v29; // [rsp+18h] [rbp-98h]
  char *v30; // [rsp+20h] [rbp-90h] BYREF
  char v31; // [rsp+40h] [rbp-70h]
  char v32; // [rsp+41h] [rbp-6Fh]
  __int64 v33[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v34; // [rsp+70h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 96);
  if ( *(_BYTE *)v4 == 61 )
    v5 = *(__int64 **)(v4 + 8);
  else
    v5 = *(__int64 **)(*(_QWORD *)(v4 - 64) + 8LL);
  v24 = (__int64 **)sub_BCE1B0(v5, *(_QWORD *)(a2 + 8));
  v6 = sub_2AAE0E0(*(_QWORD *)(a1 + 96));
  v7 = *(_QWORD *)(a2 + 904);
  v23 = v6;
  v25 = *(_BYTE *)(a1 + 104);
  v33[0] = *(_QWORD *)(a1 + 88);
  if ( v33[0] )
    sub_2AAAFA0(v33);
  sub_2BF1A90(a2, (__int64)v33);
  sub_9C6650(v33);
  v8 = *(__int64 **)(a1 + 48);
  if ( *(_BYTE *)(a1 + 106) )
  {
    v9 = v8[*(_DWORD *)(a1 + 56) - 1];
    if ( v9 )
    {
      v10 = sub_2BFB640(a2, v9, 0);
      if ( *(_BYTE *)(a1 + 105) )
      {
        v33[0] = (__int64)"reverse";
        v34 = 259;
        v10 = sub_B37000((unsigned int **)v7, v10, (__int64)v33);
      }
      v21 = v10;
      v11 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), v25);
      v12 = v21;
      v13 = v11;
      if ( v25 )
      {
        if ( v21 )
        {
          v26 = v11;
          v33[0] = (__int64)"wide.masked.load";
          v34 = 259;
          v14 = sub_ACADE0(v24);
          v15 = (_BYTE *)sub_B34C20(v7, v24, v26, v23, v21, v14, (__int64)v33);
          goto LABEL_14;
        }
        goto LABEL_17;
      }
LABEL_13:
      v33[0] = (__int64)"wide.masked.gather";
      v34 = 259;
      v15 = (_BYTE *)sub_B34D80(v7, (__int64)v24, v13, v23, v12, 0, (__int64)v33);
      goto LABEL_14;
    }
  }
  v16 = sub_2BFB640(a2, *v8, v25);
  v12 = 0;
  v13 = v16;
  if ( !v25 )
    goto LABEL_13;
LABEL_17:
  v32 = 1;
  v31 = 3;
  v30 = "wide.load";
  v22 = v13;
  v34 = 257;
  v19 = sub_BD2C40(80, 1u);
  v20 = v19;
  if ( v19 )
  {
    v28 = v19;
    sub_B4D190((__int64)v19, (__int64)v24, v22, (__int64)v33, 0, v23, 0, 0);
    v20 = v28;
  }
  v29 = (__int64)v20;
  (*(void (__fastcall **)(_QWORD, _QWORD *, char **, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
    *(_QWORD *)(v7 + 88),
    v20,
    &v30,
    *(_QWORD *)(v7 + 56),
    *(_QWORD *)(v7 + 64));
  sub_94AAF0((unsigned int **)v7, v29);
  v15 = (_BYTE *)v29;
LABEL_14:
  v27 = (__int64)v15;
  sub_2BF08A0(a2, v15, (_BYTE *)v4);
  v17 = v27;
  if ( *(_BYTE *)(a1 + 105) )
  {
    v33[0] = (__int64)"reverse";
    v34 = 259;
    v17 = sub_B37000((unsigned int **)v7, v27, (__int64)v33);
  }
  return sub_2BF26E0(a2, a1 + 112, v17, 0);
}
