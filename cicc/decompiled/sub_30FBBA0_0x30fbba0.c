// Function: sub_30FBBA0
// Address: 0x30fbba0
//
__int64 *__fastcall sub_30FBBA0(__int64 *a1, _BYTE *a2, _QWORD *a3, char a4)
{
  __int64 v6; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // [rsp+8h] [rbp-48h]
  __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  sub_30FBAE0(v11, (__int64)a2, a3);
  v6 = v11[0];
  if ( v11[0] )
    goto LABEL_2;
  if ( a4 && !a2[360] )
  {
    (*(void (__fastcall **)(__int64 *, _BYTE *, _QWORD *))(*(_QWORD *)a2 + 64LL))(v11, a2, a3);
    v6 = v11[0];
LABEL_2:
    *a1 = v6;
    return a1;
  }
  v10 = sub_30CC5F0((__int64)a2, (__int64)a3);
  v8 = sub_22077B0(0x40u);
  v9 = v8;
  if ( v8 )
    sub_30CABE0(v8, (__int64)a2, a3, v10, a4);
  *a1 = v9;
  return a1;
}
