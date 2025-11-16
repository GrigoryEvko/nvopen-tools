// Function: sub_169D620
// Address: 0x169d620
//
__int64 __fastcall sub_169D620(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rax
  bool v4; // [rsp+13h] [rbp-BDh] BYREF
  __int64 v5; // [rsp+14h] [rbp-BCh] BYREF
  int v6; // [rsp+1Ch] [rbp-B4h]
  __int16 *v7; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD *v8; // [rsp+28h] [rbp-A8h]
  _QWORD *v9; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v10; // [rsp+38h] [rbp-98h]
  __int64 v11[4]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v12[2]; // [rsp+60h] [rbp-70h] BYREF
  char v13; // [rsp+72h] [rbp-5Eh]
  __int16 *v14; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+88h] [rbp-48h]

  v2 = (_QWORD *)*a2;
  v5 = *(_QWORD *)*a2;
  v6 = *((_DWORD *)v2 + 2);
  WORD1(v5) = -1022;
  sub_16986C0(v11, a2);
  sub_16995F0((__int64)v11, (__int16 *)&v5, 0, &v4);
  sub_16986C0(v12, v11);
  sub_16995F0((__int64)v12, word_42AE9D0, 0, &v4);
  sub_169AA10((__int64)&v14, (__int64)v12);
  if ( v15 <= 0x40 )
  {
    v7 = v14;
  }
  else
  {
    v7 = *(__int16 **)v14;
    j_j___libc_free_0_0(v14);
  }
  if ( (v13 & 6) != 0 && (v13 & 7) != 3 && v4 )
  {
    sub_16995F0((__int64)v12, (__int16 *)&v5, 0, &v4);
    sub_16986C0(&v14, v11);
    sub_169D430(&v14, v12, 0);
    sub_16995F0((__int64)&v14, word_42AE9D0, 0, &v4);
    sub_169AA10((__int64)&v9, (__int64)&v14);
    if ( v10 <= 0x40 )
    {
      v8 = v9;
    }
    else
    {
      v8 = (_QWORD *)*v9;
      j_j___libc_free_0_0(v9);
    }
    sub_1698460((__int64)&v14);
  }
  else
  {
    v8 = 0;
  }
  sub_16A50F0(a1, 128, &v7, 2);
  sub_1698460((__int64)v12);
  sub_1698460((__int64)v11);
  return a1;
}
