// Function: sub_38F2290
// Address: 0x38f2290
//
__int64 __fastcall sub_38F2290(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  unsigned int v6; // ecx
  __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdi
  const char *v13; // rax
  __int64 v14[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v16; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)a1;
  v14[0] = 0;
  v14[1] = 0;
  v3 = sub_3909460(v2);
  v4 = sub_39092A0(v3);
  if ( (unsigned __int8)sub_38F0EE0(*(_QWORD *)a1, v14, v5, v6) )
  {
    v7 = *(_QWORD *)a1;
    v15[0] = "expected identifier";
    v16 = 259;
    return (unsigned int)sub_3909790(v7, v4, v15, 0, 0);
  }
  v10 = *(_QWORD *)(*(_QWORD *)a1 + 320LL);
  v15[0] = v14;
  v16 = 261;
  v11 = sub_38BF510(v10, (__int64)v15);
  v8 = *(_BYTE *)(v11 + 8) & 1;
  if ( (*(_BYTE *)(v11 + 8) & 1) != 0 )
  {
    HIBYTE(v16) = 1;
    v12 = *(_QWORD *)a1;
    v13 = "non-local symbol required";
  }
  else
  {
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)a1 + 328LL) + 256LL))(
           *(_QWORD *)(*(_QWORD *)a1 + 328LL),
           v11,
           **(unsigned int **)(a1 + 8)) )
    {
      return v8;
    }
    HIBYTE(v16) = 1;
    v12 = *(_QWORD *)a1;
    v13 = "unable to emit symbol attribute";
  }
  v15[0] = v13;
  LOBYTE(v16) = 3;
  return (unsigned int)sub_3909790(v12, v4, v15, 0, 0);
}
