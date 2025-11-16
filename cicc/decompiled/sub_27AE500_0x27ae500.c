// Function: sub_27AE500
// Address: 0x27ae500
//
__int64 __fastcall sub_27AE500(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  unsigned __int64 *v11; // r13
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rdi
  unsigned __int64 v14[2]; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE v15[32]; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int64 v16[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v17[32]; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int64 v18[2]; // [rsp+70h] [rbp-90h] BYREF
  _BYTE v19[32]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v20[2]; // [rsp+A0h] [rbp-60h] BYREF
  _BYTE v21[80]; // [rsp+B0h] [rbp-50h] BYREF

  v7 = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)v7 )
    return sub_C7D6A0(*(_QWORD *)(a1 + 8), 96 * v7, 8);
  if ( !byte_4FFC580 && (unsigned int)sub_2207590((__int64)&byte_4FFC580) )
  {
    qword_4FFC5B0 = 0;
    qword_4FFC5A0 = (__int64)&qword_4FFC5B0;
    qword_4FFC5D0 = (__int64)algn_4FFC5E0;
    qword_4FFC5D8 = 0x400000000LL;
    qword_4FFC5A8 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC5A0, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC580);
  }
  v14[0] = (unsigned __int64)v15;
  v14[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FFC5A8 )
    sub_27ABF90((__int64)v14, (__int64)&qword_4FFC5A0, a3, a4, a5, a6);
  v9 = (unsigned int)qword_4FFC5D8;
  v16[0] = (unsigned __int64)v17;
  v16[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FFC5D8 )
    sub_27AC1D0((__int64)v16, (__int64)&qword_4FFC5D0, a3, (unsigned int)qword_4FFC5D8, a5, a6);
  if ( !byte_4FFC508 && (unsigned int)sub_2207590((__int64)&byte_4FFC508) )
  {
    qword_4FFC530 = 1;
    qword_4FFC520 = (__int64)&qword_4FFC530;
    qword_4FFC550 = (__int64)algn_4FFC560;
    qword_4FFC558 = 0x400000000LL;
    qword_4FFC528 = 0x400000001LL;
    __cxa_atexit((void (*)(void *))sub_27ABC80, &qword_4FFC520, &qword_4A427C0);
    sub_2207640((__int64)&byte_4FFC508);
  }
  v10 = (unsigned int)qword_4FFC528;
  v18[0] = (unsigned __int64)v19;
  v18[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FFC528 )
    sub_27ABF90((__int64)v18, (__int64)&qword_4FFC520, (unsigned int)qword_4FFC528, v9, a5, a6);
  v20[0] = (unsigned __int64)v21;
  v20[1] = 0x400000000LL;
  if ( (_DWORD)qword_4FFC558 )
  {
    sub_27AC1D0((__int64)v20, (__int64)&qword_4FFC550, v10, v9, a5, a6);
    v11 = *(unsigned __int64 **)(a1 + 8);
    v12 = &v11[12 * *(unsigned int *)(a1 + 24)];
    if ( v11 == v12 )
    {
LABEL_18:
      if ( (_BYTE *)v20[0] != v21 )
        _libc_free(v20[0]);
      goto LABEL_20;
    }
    do
    {
LABEL_13:
      v13 = v11[6];
      if ( (unsigned __int64 *)v13 != v11 + 8 )
        _libc_free(v13);
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        _libc_free(*v11);
      v11 += 12;
    }
    while ( v11 != v12 );
    goto LABEL_18;
  }
  v11 = *(unsigned __int64 **)(a1 + 8);
  v12 = &v11[12 * *(unsigned int *)(a1 + 24)];
  if ( v11 != v12 )
    goto LABEL_13;
LABEL_20:
  if ( (_BYTE *)v18[0] != v19 )
    _libc_free(v18[0]);
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0]);
  if ( (_BYTE *)v14[0] != v15 )
    _libc_free(v14[0]);
  v7 = *(unsigned int *)(a1 + 24);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 96 * v7, 8);
}
