// Function: sub_251BFD0
// Address: 0x251bfd0
//
__int64 __fastcall sub_251BFD0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        _QWORD *a4,
        _BYTE *a5,
        char a6,
        int a7,
        char a8)
{
  unsigned int v8; // ebx
  __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r13
  __int64 (*v21)(); // rax
  unsigned __int64 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  __int64 v26; // [rsp+28h] [rbp-38h]

  v8 = *(unsigned __int8 *)(a1 + 4300);
  if ( !(_BYTE)v8 )
    return v8;
  v24 = 0;
  if ( a3 )
    v24 = *(_QWORD *)(a3 + 80);
  v12 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)(a1 + 3588) )
  {
    v13 = *(_QWORD **)(a1 + 3568);
    v14 = &v13[*(unsigned int *)(a1 + 3580)];
    if ( v13 != v14 )
    {
      while ( v12 != *v13 )
      {
        if ( v14 == ++v13 )
          goto LABEL_12;
      }
      return 0;
    }
  }
  else if ( sub_C8CA60(a1 + 3560, v12) )
  {
    return 0;
  }
LABEL_12:
  v16 = sub_B43CB0(a2);
  v17 = v16;
  if ( !a4 || (v17 = v16, v16 != sub_25096F0(a4 + 9)) )
  {
    sub_250D230((unsigned __int64 *)&v25, v17, 4, v24);
    a4 = (_QWORD *)sub_251BBC0(a1, v25, v26, a3, 2, 0, 1);
  }
  if ( !a4 || (_QWORD *)a3 == a4 )
    return 0;
  v18 = *a4;
  if ( a6 )
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, _QWORD))(v18 + 160))(a4, *(_QWORD *)(a2 + 40)) )
    {
LABEL_31:
      if ( a3 )
        sub_250ED80(a1, (__int64)a4, a3, a7);
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *, unsigned __int64))(*a4 + 144LL))(a4, a2) )
        goto LABEL_29;
      return v8;
    }
    return 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD *, unsigned __int64))(v18 + 136))(a4, a2) )
    goto LABEL_31;
  sub_250D230((unsigned __int64 *)&v25, a2, 1, v24);
  v19 = v25;
  v20 = sub_251BBC0(a1, v25, v26, a3, 2, 0, 1);
  if ( !v20 )
    return 0;
  if ( a3 == v20 )
    return 0;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v20 + 112LL))(v20, v19) )
  {
    if ( !a8 )
      return 0;
    if ( *(_BYTE *)a2 != 62 )
      return 0;
    v21 = *(__int64 (**)())(*(_QWORD *)v20 + 152LL);
    if ( v21 == sub_2505DA0 || !((unsigned __int8 (__fastcall *)(__int64))v21)(v20) )
      return 0;
  }
  if ( a3 )
    sub_250ED80(a1, v20, a3, a7);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v20 + 120LL))(v20) )
LABEL_29:
    *a5 = 1;
  return v8;
}
