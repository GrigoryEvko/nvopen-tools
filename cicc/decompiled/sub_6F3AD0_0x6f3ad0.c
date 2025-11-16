// Function: sub_6F3AD0
// Address: 0x6f3ad0
//
_DWORD *__fastcall sub_6F3AD0(__int64 a1, int a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v6; // rsi
  char v9; // bl
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r14
  _DWORD *result; // rax
  __int64 v17; // rax
  __int64 v18[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = a4;
  v9 = a4;
  v10 = sub_7D0010(a1, a4);
  v15 = v10;
  if ( a2 )
  {
    v18[0] = sub_724DC0(a1, v6, v11, v12, v13, v14);
    sub_724C70(v18[0], 12);
    sub_7249B0(v18[0], 11);
    v17 = v18[0];
    *(_QWORD *)(v18[0] + 184) = *(_QWORD *)(v15 + 88);
    *(_QWORD *)(v17 + 192) = a3;
    *(_QWORD *)(v17 + 128) = *(_QWORD *)&dword_4D03B80;
    sub_6F5800(a3);
    sub_6E6A50(v18[0], a5);
    result = (_DWORD *)sub_724E30(v18);
  }
  else
  {
    result = sub_6F35D0(v10, (_QWORD *)a5);
  }
  *(_BYTE *)(a5 + 18) = *(_BYTE *)(a5 + 18) & 0xB7 | ((v9 & 1) << 6) | 8;
  return result;
}
