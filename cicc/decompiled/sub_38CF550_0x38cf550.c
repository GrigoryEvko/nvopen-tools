// Function: sub_38CF550
// Address: 0x38cf550
//
_BYTE *__fastcall sub_38CF550(__int64 **a1, __int64 a2)
{
  _BYTE *result; // rax
  __int64 v3; // r12
  _BYTE *v4; // rax
  __int64 v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  const char *v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // [rsp-88h] [rbp-88h] BYREF
  __int64 v15; // [rsp-80h] [rbp-80h]
  char *v16; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v17; // [rsp-70h] [rbp-70h]
  __int16 v18; // [rsp-68h] [rbp-68h]
  _QWORD v19[2]; // [rsp-58h] [rbp-58h] BYREF
  __int16 v20; // [rsp-48h] [rbp-48h]
  __int64 v21; // [rsp-38h] [rbp-38h] BYREF
  __int64 v22; // [rsp-30h] [rbp-30h]
  __int64 v23; // [rsp-28h] [rbp-28h]
  int v24; // [rsp-20h] [rbp-20h]

  result = (_BYTE *)a2;
  if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
  {
    *(_BYTE *)(a2 + 8) |= 4u;
    v3 = *(_QWORD *)(a2 + 24);
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    if ( !(unsigned __int8)sub_38CF2F0(v3, (__int64)&v21, a1) )
    {
      v10 = **a1;
      v19[0] = "expression could not be evaluated";
      v20 = 259;
      sub_38BE3D0(v10, *(_QWORD *)(v3 + 8), (__int64)v19);
      return 0;
    }
    if ( v22 )
    {
      v4 = *(_BYTE **)(v22 + 24);
      v5 = **a1;
      if ( (*v4 & 4) != 0 )
      {
        v6 = (__int64 *)*((_QWORD *)v4 - 1);
        v7 = *v6;
        v8 = v6 + 2;
      }
      else
      {
        v7 = 0;
        v8 = 0;
      }
      v14 = v8;
      v16 = "symbol '";
      v15 = v7;
      v18 = 1283;
      v17 = &v14;
      v19[0] = &v16;
      v9 = "' could not be evaluated in a subtraction expression";
    }
    else
    {
      result = (_BYTE *)v21;
      if ( !v21 )
        return result;
      result = *(_BYTE **)(v21 + 24);
      if ( (result[9] & 0xC) != 0xC )
        return result;
      v5 = **a1;
      if ( (*result & 4) != 0 )
      {
        v11 = (__int64 *)*((_QWORD *)result - 1);
        v12 = *v11;
        v13 = v11 + 2;
      }
      else
      {
        v13 = 0;
        v12 = 0;
      }
      v14 = v13;
      v16 = "Common symbol '";
      v15 = v12;
      v17 = &v14;
      v19[0] = &v16;
      v9 = "' cannot be used in assignment expr";
      v18 = 1283;
    }
    v19[1] = v9;
    v20 = 770;
    sub_38BE3D0(v5, *(_QWORD *)(v3 + 8), (__int64)v19);
    return 0;
  }
  return result;
}
