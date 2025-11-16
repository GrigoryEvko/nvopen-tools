// Function: sub_3889810
// Address: 0x3889810
//
__int64 __fastcall sub_3889810(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r13d
  int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 v12; // rsi
  __int64 v13; // [rsp+0h] [rbp-90h] BYREF
  __int64 v14; // [rsp+8h] [rbp-88h]
  _QWORD v15[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v16; // [rsp+20h] [rbp-70h]
  const char **v17; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v18; // [rsp+38h] [rbp-58h]
  __int16 v19; // [rsp+40h] [rbp-50h]
  const char *v20; // [rsp+50h] [rbp-40h] BYREF
  char *v21; // [rsp+58h] [rbp-38h]
  __int16 v22; // [rsp+60h] [rbp-30h]

  v4 = a1 + 8;
  v5 = *(unsigned __int8 *)(a4 + 8);
  v13 = a2;
  v14 = a3;
  if ( (_BYTE)v5 )
  {
    v17 = (const char **)"field '";
    v18 = &v13;
    v20 = (const char *)&v17;
    v19 = 1283;
    v21 = "' cannot be specified more than once";
    v22 = 770;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v20);
  }
  v8 = sub_3887100(a1 + 8);
  v9 = v13;
  v10 = v14;
  *(_DWORD *)(a1 + 64) = v8;
  if ( v8 == 390 )
    return (unsigned int)sub_3889300(a1, v9, v10, a4);
  if ( v8 != 381 )
  {
    v20 = "expected DWARF language";
    v22 = 259;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v20);
  }
  v11 = sub_14E7AF0(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
  if ( v11 )
  {
    *(_BYTE *)(a4 + 8) = 1;
    *(_QWORD *)a4 = v11;
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  }
  else
  {
    v12 = *(_QWORD *)(a1 + 56);
    v20 = "invalid DWARF language";
    v21 = " '";
    v22 = 771;
    v17 = &v20;
    v18 = (__int64 *)(a1 + 72);
    v19 = 1026;
    v15[0] = &v17;
    v15[1] = "'";
    v16 = 770;
    return (unsigned int)sub_38814C0(v4, v12, (__int64)v15);
  }
  return v5;
}
