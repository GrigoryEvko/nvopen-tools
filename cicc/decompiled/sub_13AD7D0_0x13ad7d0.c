// Function: sub_13AD7D0
// Address: 0x13ad7d0
//
__int64 __fastcall sub_13AD7D0(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, _BYTE *a5)
{
  __int64 v6; // r13
  __int64 v7; // r15
  char v8; // r8
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-48h]

  v6 = sub_13A62B0(a4);
  v7 = sub_13AD540(a1, *a2, v6);
  v8 = sub_14560B0(v7);
  result = 0;
  if ( !v8 )
  {
    v16 = *(_QWORD *)(a1 + 8);
    v10 = sub_13A62A0(a4);
    v11 = sub_13A5B60(v16, v7, v10, 0, 0);
    v12 = sub_14806B0(*(_QWORD *)(a1 + 8), *a2, v11, 0, 0);
    *a2 = v12;
    *a2 = sub_13AD5A0(a1, v12, v6);
    v13 = sub_1480620(*(_QWORD *)(a1 + 8), v7, 0);
    v14 = sub_13AD660(a1, *a3, v6, v13);
    *a3 = v14;
    v15 = sub_13AD540(a1, v14, v6);
    result = sub_14560B0(v15);
    if ( !(_BYTE)result )
    {
      *a5 = 0;
      return 1;
    }
  }
  return result;
}
