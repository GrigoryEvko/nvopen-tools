// Function: sub_E5C930
// Address: 0xe5c930
//
__int64 __fastcall sub_E5C930(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  const char *v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // [rsp-98h] [rbp-98h] BYREF
  __int64 v15; // [rsp-90h] [rbp-90h]
  __int64 v16; // [rsp-88h] [rbp-88h]
  int v17; // [rsp-80h] [rbp-80h]
  _QWORD v18[2]; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v19; // [rsp-68h] [rbp-68h]
  __int64 v20; // [rsp-60h] [rbp-60h]
  __int16 v21; // [rsp-58h] [rbp-58h]
  _QWORD v22[4]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v23; // [rsp-28h] [rbp-28h]

  result = a2;
  if ( (*(_BYTE *)(a2 + 9) & 0x70) == 0x20 )
  {
    *(_BYTE *)(a2 + 8) |= 8u;
    v3 = *(_QWORD *)(a2 + 24);
    v14 = 0;
    v16 = 0;
    v15 = 0;
    v17 = 0;
    if ( !(unsigned __int8)sub_E81960(v3, &v14, a1) )
    {
      v10 = *a1;
      v22[0] = "expression could not be evaluated";
      v23 = 259;
      sub_E66880(v10, *(_QWORD *)(v3 + 8), v22);
      return 0;
    }
    if ( v15 )
    {
      v4 = *(_QWORD *)(v15 + 16);
      v5 = *a1;
      if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
      {
        v6 = *(__int64 **)(v4 - 8);
        v7 = *v6;
        v8 = v6 + 3;
      }
      else
      {
        v7 = 0;
        v8 = 0;
      }
      v20 = v7;
      v18[0] = "symbol '";
      v21 = 1283;
      v19 = v8;
      v22[0] = v18;
      v9 = "' could not be evaluated in a subtraction expression";
    }
    else
    {
      result = v14;
      if ( !v14 )
        return result;
      result = *(_QWORD *)(v14 + 16);
      if ( (((*(_BYTE *)(result + 9) & 0x70) - 48) & 0xE0) != 0 )
        return result;
      v5 = *a1;
      if ( (*(_BYTE *)(result + 8) & 1) != 0 )
      {
        v11 = *(__int64 **)(result - 8);
        v12 = *v11;
        v13 = v11 + 3;
      }
      else
      {
        v13 = 0;
        v12 = 0;
      }
      v19 = v13;
      v21 = 1283;
      v22[0] = v18;
      v9 = "' cannot be used in assignment expr";
      v18[0] = "Common symbol '";
      v20 = v12;
    }
    v22[2] = v9;
    v23 = 770;
    sub_E66880(v5, *(_QWORD *)(v3 + 8), v22);
    return 0;
  }
  return result;
}
