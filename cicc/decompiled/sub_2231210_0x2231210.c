// Function: sub_2231210
// Address: 0x2231210
//
__int64 __fastcall sub_2231210(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6, char a7, char a8)
{
  _BYTE *v10; // rbp
  __int64 v11; // rax
  __int64 v12; // r9
  char v13; // al
  size_t v14; // rbp
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 (__fastcall *v18)(__int64, unsigned int); // rdx
  _QWORD *v19; // [rsp+8h] [rbp-D0h]
  __int64 v20; // [rsp+8h] [rbp-D0h]
  char v21; // [rsp+1Ch] [rbp-BCh] BYREF
  char v22; // [rsp+1Dh] [rbp-BBh]
  char v23; // [rsp+1Eh] [rbp-BAh]
  char v24; // [rsp+1Fh] [rbp-B9h]
  char s[184]; // [rsp+20h] [rbp-B8h] BYREF

  v19 = (_QWORD *)(a4 + 208);
  v10 = (_BYTE *)sub_222F790((_QWORD *)(a4 + 208), a2);
  v11 = sub_22311C0(v19, a2);
  v12 = v11;
  if ( v10[56] )
  {
    v13 = v10[94];
  }
  else
  {
    v20 = v11;
    sub_2216D60((__int64)v10);
    v12 = v20;
    v18 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v10 + 48LL);
    v13 = 37;
    if ( v18 != sub_CE72A0 )
    {
      v13 = ((__int64 (__fastcall *)(_BYTE *, __int64, __int64 (__fastcall *)(__int64, unsigned int), __int64, __int64, __int64))v18)(
              v10,
              37,
              v18,
              v16,
              v17,
              v20);
      v12 = v20;
    }
  }
  v21 = v13;
  if ( a8 )
  {
    v22 = a8;
    v23 = a7;
    v24 = 0;
  }
  else
  {
    v22 = a7;
    v23 = 0;
  }
  sub_2255E10(v12, s, 128, &v21, a6);
  v14 = strlen(s);
  if ( !a3 )
    (*(__int64 (__fastcall **)(__int64, char *, size_t))(*(_QWORD *)a2 + 96LL))(a2, s, v14);
  return a2;
}
