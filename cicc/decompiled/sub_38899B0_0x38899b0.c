// Function: sub_38899B0
// Address: 0x38899b0
//
__int64 __fastcall sub_38899B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r13d
  int v8; // eax
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned int v11; // r9d
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned __int64 v15; // rsi
  __int64 v16; // [rsp+0h] [rbp-90h] BYREF
  __int64 v17; // [rsp+8h] [rbp-88h]
  _QWORD v18[2]; // [rsp+10h] [rbp-80h] BYREF
  __int16 v19; // [rsp+20h] [rbp-70h]
  const char **v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v21; // [rsp+38h] [rbp-58h]
  __int16 v22; // [rsp+40h] [rbp-50h]
  const char *v23; // [rsp+50h] [rbp-40h] BYREF
  char *v24; // [rsp+58h] [rbp-38h]
  __int16 v25; // [rsp+60h] [rbp-30h]

  v4 = a1 + 8;
  v5 = *(unsigned __int8 *)(a4 + 8);
  v16 = a2;
  v17 = a3;
  if ( (_BYTE)v5 )
  {
    v20 = (const char **)"field '";
    v21 = &v16;
    v23 = (const char *)&v20;
    v22 = 1283;
    v24 = "' cannot be specified more than once";
    v25 = 770;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v23);
  }
  v8 = sub_3887100(a1 + 8);
  v12 = v16;
  v13 = v17;
  *(_DWORD *)(a1 + 64) = v8;
  if ( v8 == 390 )
    return (unsigned int)sub_3889300(a1, v12, v13, a4);
  if ( v8 != 378 )
  {
    v23 = "expected DWARF tag";
    v25 = 259;
    return (unsigned int)sub_38814C0(v4, *(_QWORD *)(a1 + 56), (__int64)&v23);
  }
  v14 = sub_14E0B10(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), v13, v9, v10, v11);
  if ( v14 == -1 )
  {
    v15 = *(_QWORD *)(a1 + 56);
    v23 = "invalid DWARF tag";
    v24 = " '";
    v25 = 771;
    v20 = &v23;
    v21 = (__int64 *)(a1 + 72);
    v22 = 1026;
    v18[0] = &v20;
    v18[1] = "'";
    v19 = 770;
    return (unsigned int)sub_38814C0(v4, v15, (__int64)v18);
  }
  else
  {
    *(_BYTE *)(a4 + 8) = 1;
    *(_QWORD *)a4 = v14;
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  }
  return v5;
}
