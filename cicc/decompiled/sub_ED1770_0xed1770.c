// Function: sub_ED1770
// Address: 0xed1770
//
_QWORD *__fastcall sub_ED1770(__int64 *a1, int a2, char *a3, unsigned __int64 a4)
{
  int v6; // ebx
  _QWORD *v7; // r12
  __int64 v9; // [rsp+0h] [rbp-A0h]
  _QWORD *v10; // [rsp+8h] [rbp-98h]
  __int64 v11; // [rsp+18h] [rbp-88h]
  __int64 v12[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v13; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v14; // [rsp+40h] [rbp-60h] BYREF
  __int16 v15; // [rsp+60h] [rbp-40h]

  v6 = a2;
  if ( sub_ED1700((__int64)a1) )
  {
    v6 = 0;
  }
  else
  {
    switch ( a2 )
    {
      case 9:
        v6 = 2;
        break;
      case 1:
        v6 = 3;
        break;
      case 7:
      case 0:
        v6 = 8;
        break;
    }
  }
  v9 = sub_AC9B20(*a1, a3, a4, 0);
  v10 = *(_QWORD **)(v9 + 8);
  sub_ED1630(v12, (__int64)a3, a4, v6);
  v14 = v12;
  v15 = 260;
  BYTE4(v11) = 0;
  v7 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v7 )
    sub_B30000((__int64)v7, (__int64)a1, v10, 1, v6, v9, (__int64)&v14, 0, 0, v11, 0);
  if ( (__int64 *)v12[0] != &v13 )
    j_j___libc_free_0(v12[0], v13 + 1);
  sub_ED1710((__int64)a1, (__int64)v7);
  return v7;
}
