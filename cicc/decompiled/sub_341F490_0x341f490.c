// Function: sub_341F490
// Address: 0x341f490
//
_BYTE *__fastcall sub_341F490(__int64 *a1, __int64 *a2, __int64 a3, char a4)
{
  __int128 v7; // rax
  __int128 v8; // rcx
  __int8 *v9[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v10; // [rsp+10h] [rbp-D0h] BYREF
  _OWORD v11[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v12; // [rsp+40h] [rbp-A0h]
  _QWORD v13[4]; // [rsp+50h] [rbp-90h] BYREF
  char v14; // [rsp+70h] [rbp-70h]
  char v15; // [rsp+71h] [rbp-6Fh]
  _OWORD v16[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v17; // [rsp+A0h] [rbp-40h]

  if ( !*(_QWORD *)(a3 + 24) || a4 )
  {
    v15 = 1;
    v13[0] = ")";
    v14 = 3;
    *(_QWORD *)&v7 = sub_2E791E0(a1);
    v11[1] = v7;
    LOWORD(v12) = 1283;
    *(_QWORD *)&v11[0] = " (in function: ";
    *((_QWORD *)&v8 + 1) = v13[1];
    *(_QWORD *)&v8 = ")";
    v16[1] = v8;
    *(_QWORD *)&v16[0] = v11;
    LOWORD(v17) = 770;
    sub_CA0F50((__int64 *)v9, (void **)v16);
    sub_B18290(a3, v9[0], (size_t)v9[1]);
    if ( (__int64 *)v9[0] != &v10 )
      j_j___libc_free_0((unsigned __int64)v9[0]);
    if ( a4 )
    {
      sub_B17B60((__int64)v13, a3);
      *(_QWORD *)&v16[0] = v13;
      LOWORD(v17) = 260;
      sub_C64D30((__int64)v16, 1u);
    }
  }
  return sub_1049740(a2, a3);
}
