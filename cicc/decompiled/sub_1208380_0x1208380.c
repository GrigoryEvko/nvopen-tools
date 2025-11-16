// Function: sub_1208380
// Address: 0x1208380
//
__int64 __fastcall sub_1208380(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // ebx
  unsigned __int64 v10; // rsi
  _QWORD v11[4]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v12; // [rsp+20h] [rbp-70h]
  _QWORD v13[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v14; // [rsp+50h] [rbp-40h]

  v6 = a1 + 176;
  v7 = *(unsigned __int8 *)(a4 + 8);
  if ( (_BYTE)v7 )
  {
    v11[2] = a2;
    v10 = *(_QWORD *)(a1 + 232);
    v12 = 1283;
    v11[0] = "field '";
    v11[3] = a3;
    v13[0] = v11;
    v14 = 770;
    v13[2] = "' cannot be specified more than once";
    sub_11FD800(v6, v10, (__int64)v13, 1);
    return v7;
  }
  else
  {
    *(_DWORD *)(a1 + 240) = sub_1205200(v6);
    return sub_1208110(a1, a2, a3, a4);
  }
}
