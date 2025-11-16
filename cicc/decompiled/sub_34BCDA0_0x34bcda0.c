// Function: sub_34BCDA0
// Address: 0x34bcda0
//
void __fastcall sub_34BCDA0(
        __int64 a1,
        unsigned __int8 (__fastcall *a2)(__int64, __int64 *, unsigned __int64 *),
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  _QWORD *i; // rax
  __int64 v8; // r13
  _QWORD *v9; // rdx
  __int64 v10; // r12
  __int64 j; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 (*v15)(void); // rax
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 (*v18)(); // rax
  __int64 v19; // [rsp+20h] [rbp-130h] BYREF
  __int64 v20; // [rsp+28h] [rbp-128h] BYREF
  _QWORD *v21; // [rsp+30h] [rbp-120h] BYREF
  __int64 v22; // [rsp+38h] [rbp-118h]
  _QWORD v23[6]; // [rsp+40h] [rbp-110h] BYREF
  _BYTE *v24; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v25; // [rsp+78h] [rbp-D8h]
  _BYTE v26[208]; // [rsp+80h] [rbp-D0h] BYREF

  i = v23;
  v8 = (__int64)(*(_QWORD *)(a1 + 104) - *(_QWORD *)(a1 + 96)) >> 3;
  v21 = v23;
  v22 = 0x600000000LL;
  if ( (_DWORD)v8 )
  {
    if ( (unsigned int)v8 > 6uLL )
    {
      sub_C8D5F0((__int64)&v21, v23, (unsigned int)v8, 8u, a5, a6);
      v9 = &v21[(unsigned int)v8];
      for ( i = &v21[(unsigned int)v22]; v9 != i; ++i )
      {
LABEL_4:
        if ( i )
          *i = 0;
      }
    }
    else
    {
      v9 = &v23[(unsigned int)v8];
      if ( v23 != v9 )
        goto LABEL_4;
    }
    LODWORD(v22) = v8;
  }
  v10 = *(_QWORD *)(a1 + 328);
  for ( j = a1 + 320; j != v10; v10 = *(_QWORD *)(v10 + 8) )
  {
    v12 = sub_2E32300((__int64 *)v10, 0);
    v21[*(int *)(v10 + 24)] = v12;
  }
  v13 = a3;
  v14 = 0;
  sub_34BCCA0((unsigned __int64 *)(a1 + 320), a2, v13);
  sub_2E79040(a1);
  v15 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 128LL);
  if ( v15 != sub_2DAC790 )
    v14 = v15();
  v16 = *(_QWORD *)(a1 + 328);
  v24 = v26;
  v25 = 0x400000000LL;
  if ( j != v16 )
  {
    while ( 1 )
    {
      v17 = v21[*(int *)(v16 + 24)];
      if ( !v17 )
        goto LABEL_20;
      if ( !*(_BYTE *)(v16 + 261) && v17 == *(_QWORD *)(v16 + 8) )
      {
LABEL_21:
        v20 = 0;
        LODWORD(v25) = 0;
        v19 = 0;
        v18 = *(__int64 (**)())(*(_QWORD *)v14 + 344LL);
        if ( v18 == sub_2DB1AE0
          || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v18)(
               v14,
               v16,
               &v19,
               &v20,
               &v24,
               0) )
        {
          goto LABEL_14;
        }
        sub_2E32A60(v16, v17);
        v16 = *(_QWORD *)(v16 + 8);
        if ( j == v16 )
        {
LABEL_24:
          if ( v24 != v26 )
            _libc_free((unsigned __int64)v24);
          break;
        }
      }
      else
      {
        sub_2E32880(&v20, v16);
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v14 + 368LL))(
          v14,
          v16,
          v17,
          0,
          0,
          0,
          &v20,
          0);
        if ( v20 )
          sub_B91220((__int64)&v20, v20);
LABEL_20:
        if ( !*(_BYTE *)(v16 + 261) )
          goto LABEL_21;
LABEL_14:
        v16 = *(_QWORD *)(v16 + 8);
        if ( j == v16 )
          goto LABEL_24;
      }
    }
  }
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
}
