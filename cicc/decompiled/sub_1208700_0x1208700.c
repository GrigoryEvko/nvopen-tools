// Function: sub_1208700
// Address: 0x1208700
//
__int64 __fastcall sub_1208700(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // ebx
  unsigned __int64 v7; // rsi
  int v10; // eax
  unsigned __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned __int64 v13; // rsi
  __int64 v14; // [rsp+8h] [rbp-C8h]
  _QWORD v15[4]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v16; // [rsp+30h] [rbp-A0h]
  _QWORD v17[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v18; // [rsp+50h] [rbp-80h]
  __int64 v19; // [rsp+58h] [rbp-78h]
  __int16 v20; // [rsp+60h] [rbp-70h]
  _QWORD v21[2]; // [rsp+70h] [rbp-60h] BYREF
  char *v22; // [rsp+80h] [rbp-50h]
  __int16 v23; // [rsp+90h] [rbp-40h]

  v6 = *(unsigned __int8 *)(a4 + 8);
  if ( (_BYTE)v6 )
  {
    v19 = a3;
    v20 = 1283;
    v7 = *(_QWORD *)(a1 + 232);
    v17[0] = "field '";
    v21[0] = v17;
    v23 = 770;
    v18 = a2;
    v22 = "' cannot be specified more than once";
    sub_11FD800(a1 + 176, v7, (__int64)v21, 1);
    return v6;
  }
  v14 = a1 + 176;
  v10 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v10;
  if ( v10 != 529 )
  {
    if ( v10 == 516 )
    {
      v12 = sub_E0AB40(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
      if ( v12 )
      {
        *(_BYTE *)(a4 + 8) = 1;
        *(_QWORD *)a4 = v12;
        *(_DWORD *)(a1 + 240) = sub_1205200(v14);
      }
      else
      {
        v13 = *(_QWORD *)(a1 + 232);
        v21[0] = "invalid DWARF language";
        v6 = 1;
        v22 = " '";
        v23 = 771;
        v17[0] = v21;
        v18 = a1 + 248;
        v20 = 1026;
        v15[0] = v17;
        v16 = 770;
        v15[2] = "'";
        sub_11FD800(v14, v13, (__int64)v15, 1);
      }
    }
    else
    {
      v23 = 259;
      v6 = 1;
      v11 = *(_QWORD *)(a1 + 232);
      v21[0] = "expected DWARF language";
      sub_11FD800(v14, v11, (__int64)v21, 1);
    }
    return v6;
  }
  return sub_1208110(a1, a2, a3, a4);
}
