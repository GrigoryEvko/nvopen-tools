// Function: sub_1233170
// Address: 0x1233170
//
__int64 __fastcall sub_1233170(__int64 a1, unsigned __int8 **a2, __int64 *a3)
{
  unsigned int v3; // r13d
  unsigned __int64 v5; // r14
  int v6; // eax
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v13; // [rsp+8h] [rbp-68h] BYREF
  const char *v14[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v15; // [rsp+30h] [rbp-40h]

  v13 = 0;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v12, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after vaarg operand") )
    return 1;
  v5 = *(_QWORD *)(a1 + 232);
  v14[0] = "expected type";
  v15 = 259;
  v3 = sub_12190A0(a1, &v13, (int *)v14, 0);
  if ( (_BYTE)v3 )
    return 1;
  v6 = *((unsigned __int8 *)v13 + 8);
  if ( v6 == 13 || v6 == 7 )
  {
    v15 = 259;
    v14[0] = "va_arg requires operand with first class type";
    sub_11FD800(a1 + 176, v5, (__int64)v14, 1);
    return 1;
  }
  v15 = 257;
  v7 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
  v8 = v7;
  if ( v7 )
  {
    v9 = v12;
    sub_B44260((__int64)v7, (__int64)v13, 60, 1u, 0, 0);
    if ( *((_QWORD *)v8 - 4) )
    {
      v10 = *((_QWORD *)v8 - 3);
      **((_QWORD **)v8 - 2) = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *((_QWORD *)v8 - 2);
    }
    *((_QWORD *)v8 - 4) = v9;
    if ( v9 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      *((_QWORD *)v8 - 3) = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = v8 - 24;
      *((_QWORD *)v8 - 2) = v9 + 16;
      *(_QWORD *)(v9 + 16) = v8 - 32;
    }
    sub_BD6B50(v8, v14);
  }
  *a2 = v8;
  return v3;
}
