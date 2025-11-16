// Function: sub_CA8B70
// Address: 0xca8b70
//
char *__fastcall sub_CA8B70(char *a1, unsigned __int64 a2, _QWORD *a3)
{
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rsi
  char *v7; // r12
  char *v9; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v10; // [rsp+8h] [rbp-48h]
  _BYTE v11[16]; // [rsp+10h] [rbp-40h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-30h]

  v9 = a1;
  v10 = a2;
  v5 = sub_C93740((__int64 *)&v9, "\r\n \t", 4, 0xFFFFFFFFFFFFFFFFLL) + 1;
  v12 = 0;
  if ( v5 > v10 )
    v5 = v10;
  v6 = v10 - a2 + v5;
  if ( v6 > v10 )
    v6 = v10;
  v10 = v6;
  v7 = sub_CA67E0(v9, v6, a3, (unsigned __int8 *)"\r\n", 2, (__int64)v11);
  if ( v12 )
    v12(v11, v11, 3);
  return v7;
}
