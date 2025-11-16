// Function: sub_1FDC040
// Address: 0x1fdc040
//
__int64 __fastcall sub_1FDC040(
        __int64 **a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned __int8 a5,
        unsigned __int64 a6,
        unsigned __int8 a7)
{
  unsigned int v7; // r13d
  unsigned __int64 v8; // r12
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 (*v12)(); // r11
  __int64 (*v13)(); // rax
  _QWORD *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 (*v19)(); // r11
  __int64 result; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  unsigned int v23; // [rsp+8h] [rbp-38h]
  unsigned int v24; // [rsp+Ch] [rbp-34h]
  int v25; // [rsp+Ch] [rbp-34h]
  unsigned int v26; // [rsp+Ch] [rbp-34h]

  v7 = a3;
  v8 = a6;
  if ( a3 == 54 )
  {
    if ( !a6 || (a6 & (a6 - 1)) != 0 )
      goto LABEL_5;
    _BitScanReverse64(&v22, a6);
    v7 = 122;
    v8 = 63 - ((unsigned int)v22 ^ 0x3F);
  }
  else if ( a3 == 56 )
  {
    if ( !a6 || (a6 & (a6 - 1)) != 0 )
      goto LABEL_5;
    _BitScanReverse64(&v21, a6);
    v7 = 124;
    v8 = 63 - ((unsigned int)v21 ^ 0x3F);
  }
  else if ( a3 - 122 > 2 )
  {
    goto LABEL_5;
  }
  if ( (unsigned int)sub_1FD3510(a2) <= v8 )
    return 0;
LABEL_5:
  v10 = *a1;
  LODWORD(v11) = a5;
  v12 = (__int64 (*)())(*a1)[10];
  if ( v12 == sub_1FD3500 )
    goto LABEL_6;
  v25 = a5;
  result = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, _QWORD, _QWORD))v12)(a1, a2, a2, v7, a4);
  LODWORD(v11) = v25;
  if ( !(_DWORD)result )
  {
    v10 = *a1;
LABEL_6:
    v13 = (__int64 (*)())v10[11];
    if ( v13 != sub_1FD34E0 )
    {
      v26 = v11;
      v17 = ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, __int64, unsigned __int64))v13)(a1, a7, a7, 10, v8);
      v11 = v26;
      v18 = 1;
      if ( (_DWORD)v17 )
      {
LABEL_9:
        v19 = (__int64 (*)())(*a1)[9];
        if ( v19 != sub_1FD34D0 )
          return ((__int64 (__fastcall *)(__int64 **, _QWORD, _QWORD, _QWORD, _QWORD, __int64, __int64, __int64))v19)(
                   a1,
                   a2,
                   a2,
                   v7,
                   a4,
                   v11,
                   v17,
                   v18);
        return 0;
      }
    }
    v23 = v11;
    v24 = sub_1FD3510(a2);
    v14 = (_QWORD *)sub_15E0530(*a1[5]);
    v15 = sub_1644900(v14, v24);
    v16 = sub_159C470(v15, v8, 0);
    v17 = sub_1FD8F60(a1, v16);
    if ( (_DWORD)v17 )
    {
      v11 = v23;
      v18 = 0;
      goto LABEL_9;
    }
    return 0;
  }
  return result;
}
