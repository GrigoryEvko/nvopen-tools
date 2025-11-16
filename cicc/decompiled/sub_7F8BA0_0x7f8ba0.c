// Function: sub_7F8BA0
// Address: 0x7f8ba0
//
void __fastcall sub_7F8BA0(__int64 a1, char a2, int *a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  char v9; // bl
  __int64 v10; // rsi
  _BYTE *v11; // rax
  _BYTE *v12; // rbx
  _BYTE *v13; // rax
  char v14; // si
  char v15; // bl
  _BYTE *v16; // rax
  _BYTE *v17; // [rsp+8h] [rbp-48h]
  __int64 *v19; // [rsp+18h] [rbp-38h]
  _BYTE *v20; // [rsp+18h] [rbp-38h]

  v9 = a2 & 1;
  if ( (unsigned int)(*a3 - 3) > 2 )
  {
    v13 = sub_726B30(1);
    v14 = 2 * v9;
    v15 = v13[41];
    *((_QWORD *)v13 + 6) = a1;
    v20 = v13;
    v13[41] = v14 | v15 & 0xFD;
    sub_7E6810((__int64)v13, (__int64)a3, 1);
    v12 = sub_726B30(11);
    *((_QWORD *)v20 + 9) = v12;
    sub_7E1740((__int64)v12, a5);
    if ( a6 )
    {
      v16 = sub_726B30(11);
      *((_QWORD *)v20 + 10) = v16;
      sub_7E1740((__int64)v16, a6);
    }
  }
  else
  {
    v19 = (__int64 *)sub_7F8B70();
    v17 = sub_7F8B70();
    *(_QWORD *)(a1 + 16) = v19;
    v10 = *v19;
    v19[2] = (__int64)v17;
    v11 = sub_73DBF0(0x67u, v10, a1);
    v11[25] = (8 * v9) | v11[25] & 0xF7;
    v12 = 0;
    sub_7E25D0((__int64)v11, a3);
    sub_7E1780((__int64)v19, a5);
    if ( a6 )
      sub_7E1780((__int64)v17, a6);
  }
  if ( a4 )
    *a4 = v12;
}
