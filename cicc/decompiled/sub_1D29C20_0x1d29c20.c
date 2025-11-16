// Function: sub_1D29C20
// Address: 0x1d29c20
//
_QWORD *__fastcall sub_1D29C20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // r13
  int v9; // ebx
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int v14; // ebx
  int v15; // r13d
  __int64 v16; // rax
  unsigned int v17; // edx
  unsigned __int8 v18; // al
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h]

  v6 = a4;
  v7 = a1[4];
  v20 = a2;
  v21 = a3;
  v8 = *(_QWORD *)(v7 + 56);
  if ( (_BYTE)a2 )
    v9 = sub_1D13440(a2);
  else
    v9 = sub_1F58D40(&v20, a2, a3, a4, a5, a6);
  v10 = sub_1F58E60(&v20, a1[6]);
  v11 = sub_1E0A0C0(a1[4]);
  v12 = sub_15AAE50(v11, v10);
  v13 = v6;
  v14 = v9 + 7;
  if ( v12 >= v6 )
    v13 = v12;
  v15 = sub_1E090F0(v8, v14 >> 3, v13, 0, 0, 0, v20, v21);
  v16 = sub_1E0A0C0(a1[4]);
  v17 = 8 * sub_15A9520(v16, *(_DWORD *)(v16 + 4));
  if ( v17 == 32 )
  {
    v18 = 5;
  }
  else if ( v17 > 0x20 )
  {
    v18 = 6;
    if ( v17 != 64 )
    {
      v18 = 0;
      if ( v17 == 128 )
        v18 = 7;
    }
  }
  else
  {
    v18 = 3;
    if ( v17 != 8 )
      v18 = 4 * (v17 == 16);
  }
  return sub_1D299D0(a1, v15, v18, 0, 0);
}
