// Function: sub_E9A6D0
// Address: 0xe9a6d0
//
__int64 __fastcall sub_E9A6D0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 (*v5)(); // rax
  char v6; // al
  _QWORD *v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // rdi
  _QWORD *v11; // rsi
  unsigned __int8 v12; // [rsp+Fh] [rbp-91h]
  unsigned __int64 v13; // [rsp+18h] [rbp-88h] BYREF
  _QWORD v14[4]; // [rsp+20h] [rbp-80h] BYREF
  __int16 v15; // [rsp+40h] [rbp-60h]
  _QWORD v16[4]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v17; // [rsp+70h] [rbp-30h]

  v13 = 0;
  if ( !a3 )
    goto LABEL_6;
  v4 = 0;
  v5 = *(__int64 (**)())(*a1 + 80LL);
  if ( v5 != sub_C13ED0 )
    v4 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))v5)(a1, a2, 0);
  v6 = sub_E81930(a3, &v13, v4);
  if ( v6 )
  {
    if ( v13 <= 0x7FFFFFFF )
    {
LABEL_6:
      (*(void (__fastcall **)(_QWORD *, __int64))(*a1 + 176LL))(a1, a2);
      return 0;
    }
    v12 = v6;
    v8 = *(_QWORD **)(a3 + 8);
    v14[0] = "subsection number ";
    v9 = a1[1];
    v16[0] = v14;
    v15 = 3075;
    v17 = 770;
    v14[2] = &v13;
    v16[2] = " is not within [0,2147483647]";
    sub_E66880(v9, v8, (__int64)v16);
    return v12;
  }
  else
  {
    v10 = a1[1];
    v11 = *(_QWORD **)(a3 + 8);
    v17 = 259;
    v16[0] = "cannot evaluate subsection number";
    sub_E66880(v10, v11, (__int64)v16);
    return 1;
  }
}
