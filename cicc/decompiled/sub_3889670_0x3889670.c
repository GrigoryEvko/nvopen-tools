// Function: sub_3889670
// Address: 0x3889670
//
__int64 __fastcall sub_3889670(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r13d
  int v8; // eax
  __int64 v9; // rcx
  unsigned int v10; // r8d
  __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned int v13; // eax
  unsigned __int64 v14; // rsi
  __int64 v15; // [rsp+0h] [rbp-90h] BYREF
  __int64 v16; // [rsp+8h] [rbp-88h]
  _QWORD v17[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v18; // [rsp+20h] [rbp-70h]
  const char **v19; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v20; // [rsp+38h] [rbp-58h]
  __int16 v21; // [rsp+40h] [rbp-50h]
  const char *v22; // [rsp+50h] [rbp-40h] BYREF
  char *v23; // [rsp+58h] [rbp-38h]
  __int16 v24; // [rsp+60h] [rbp-30h]

  v4 = a1 + 8;
  v5 = *(unsigned __int8 *)(a4 + 8);
  v15 = a2;
  v16 = a3;
  if ( (_BYTE)v5 )
  {
    v19 = (const char **)"field '";
    v20 = &v15;
    v22 = (const char *)&v19;
    v21 = 1283;
    v23 = "' cannot be specified more than once";
    v24 = 770;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v22);
  }
  v8 = sub_3887100(a1 + 8);
  v11 = v15;
  v12 = v16;
  *(_DWORD *)(a1 + 64) = v8;
  if ( v8 == 390 )
    return (unsigned int)sub_3889300(a1, v11, v12, a4);
  if ( v8 != 379 )
  {
    v22 = "expected DWARF type attribute encoding";
    v24 = 259;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v22);
  }
  v13 = sub_14E7070(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), v12, v9, v10);
  if ( v13 )
  {
    *(_BYTE *)(a4 + 8) = 1;
    *(_QWORD *)a4 = v13;
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  }
  else
  {
    v14 = *(_QWORD *)(a1 + 56);
    v22 = "invalid DWARF type attribute encoding";
    v23 = " '";
    v24 = 771;
    v19 = &v22;
    v20 = (__int64 *)(a1 + 72);
    v21 = 1026;
    v17[0] = &v19;
    v17[1] = "'";
    v18 = 770;
    return (unsigned int)sub_38814C0(v4, v14, (__int64)v17);
  }
  return v5;
}
