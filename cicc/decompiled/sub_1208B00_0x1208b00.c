// Function: sub_1208B00
// Address: 0x1208b00
//
__int64 __fastcall sub_1208B00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // ebx
  unsigned __int64 v6; // rsi
  int v9; // eax
  unsigned int v10; // eax
  _QWORD v12[4]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v13; // [rsp+30h] [rbp-A0h]
  _QWORD v14[2]; // [rsp+40h] [rbp-90h] BYREF
  char *v15; // [rsp+50h] [rbp-80h]
  __int64 v16; // [rsp+58h] [rbp-78h]
  __int16 v17; // [rsp+60h] [rbp-70h]
  _QWORD v18[2]; // [rsp+70h] [rbp-60h] BYREF
  char *v19; // [rsp+80h] [rbp-50h]
  __int16 v20; // [rsp+90h] [rbp-40h]

  v4 = a1 + 176;
  v5 = *(unsigned __int8 *)(a4 + 8);
  if ( (_BYTE)v5 )
  {
    v14[0] = "field '";
    v17 = 1283;
    v6 = *(_QWORD *)(a1 + 232);
    v15 = "tag";
    v18[0] = v14;
    v20 = 770;
    v16 = 3;
    v19 = "' cannot be specified more than once";
    sub_11FD800(a1 + 176, v6, (__int64)v18, 1);
    return v5;
  }
  v9 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v9;
  if ( v9 != 529 )
  {
    if ( v9 == 513 )
    {
      v10 = sub_E03450(*(_BYTE **)(a1 + 248), *(_QWORD *)(a1 + 256));
      if ( v10 != -1 )
      {
        *(_BYTE *)(a4 + 8) = 1;
        *(_QWORD *)a4 = v10;
        *(_DWORD *)(a1 + 240) = sub_1205200(v4);
        return v5;
      }
      v12[0] = "invalid DWARF tag";
      v12[2] = " '";
      v13 = 771;
      v14[0] = v12;
      v15 = (char *)(a1 + 248);
      v18[0] = v14;
      v17 = 1026;
      v19 = "'";
      v20 = 770;
    }
    else
    {
      v18[0] = "expected DWARF tag";
      v20 = 259;
    }
    v5 = 1;
    sub_11FD800(v4, *(_QWORD *)(a1 + 232), (__int64)v18, 1);
    return v5;
  }
  return sub_1208110(a1, a2, a3, a4);
}
