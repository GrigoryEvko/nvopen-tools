// Function: sub_F7CA10
// Address: 0xf7ca10
//
_QWORD *__fastcall sub_F7CA10(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _QWORD *v7; // rdi
  __int64 v8; // r14
  _QWORD *v9; // r12
  __int64 v11; // r10
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rdx
  int v17; // eax
  char v18; // al
  int v19; // edx
  __int64 v22[2]; // [rsp+18h] [rbp-78h] BYREF
  __int64 v23; // [rsp+28h] [rbp-68h]
  _BYTE v24[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v25; // [rsp+50h] [rbp-40h]

  v7 = (_QWORD *)a1[9];
  v22[0] = a3;
  v8 = sub_BCB2B0(v7);
  v9 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *, __int64, _QWORD))(*(_QWORD *)a1[10]
                                                                                                  + 64LL))(
                   a1[10],
                   v8,
                   a2,
                   v22,
                   1,
                   a5);
  if ( v9 )
    return v9;
  v25 = 257;
  v9 = sub_BD2C40(88, 2u);
  if ( v9 )
  {
    v11 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    {
LABEL_5:
      sub_B44260((__int64)v9, v11, 34, 2u, 0, 0);
      v9[9] = v8;
      v9[10] = sub_B4DC50(v8, (__int64)v22, 1);
      sub_B4D9A0((__int64)v9, a2, v22, 1, (__int64)v24);
      goto LABEL_6;
    }
    v16 = *(_QWORD *)(v22[0] + 8);
    v17 = *(unsigned __int8 *)(v16 + 8);
    if ( v17 == 17 )
    {
      v18 = 0;
    }
    else
    {
      if ( v17 != 18 )
        goto LABEL_5;
      v18 = 1;
    }
    v19 = *(_DWORD *)(v16 + 32);
    BYTE4(v23) = v18;
    LODWORD(v23) = v19;
    v11 = sub_BCE1B0((__int64 *)v11, v23);
    goto LABEL_5;
  }
LABEL_6:
  sub_B4DDE0((__int64)v9, a5);
  (*(void (__fastcall **)(__int64, _QWORD *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a4,
    a1[7],
    a1[8]);
  v12 = *a1;
  v13 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v13 )
  {
    do
    {
      v14 = *(_QWORD *)(v12 + 8);
      v15 = *(_DWORD *)v12;
      v12 += 16;
      sub_B99FD0((__int64)v9, v15, v14);
    }
    while ( v13 != v12 );
  }
  return v9;
}
