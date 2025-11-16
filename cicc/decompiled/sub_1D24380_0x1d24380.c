// Function: sub_1D24380
// Address: 0x1d24380
//
_QWORD *__fastcall sub_1D24380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, __int64 *a7, int a8)
{
  __int64 v8; // r10
  __int64 v14; // rsi
  _QWORD *v15; // rax
  unsigned __int8 *v16; // rsi
  _QWORD *v17; // r8
  int v18; // eax
  _QWORD *v20; // [rsp+8h] [rbp-48h]
  _QWORD v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = a1;
  v14 = *a7;
  v21[0] = v14;
  if ( v14 )
  {
    sub_1623A60((__int64)v21, v14, 2);
    v8 = a1;
  }
  v15 = (_QWORD *)sub_145CBF0(*(__int64 **)(v8 + 648), 56, 16);
  v16 = (unsigned __int8 *)v21[0];
  v15[2] = a2;
  v17 = v15;
  v15[3] = a3;
  v15[4] = v16;
  if ( v16 )
  {
    v20 = v15;
    sub_1623210((__int64)v21, v16, (__int64)(v15 + 4));
    v18 = a8;
    v17 = v20;
  }
  else
  {
    v18 = a8;
  }
  *((_DWORD *)v17 + 10) = v18;
  *((_BYTE *)v17 + 48) = a6;
  *((_BYTE *)v17 + 49) = 0;
  *((_DWORD *)v17 + 11) = 0;
  *v17 = a4;
  *((_DWORD *)v17 + 2) = a5;
  return v17;
}
