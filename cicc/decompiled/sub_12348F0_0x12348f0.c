// Function: sub_12348F0
// Address: 0x12348f0
//
__int64 __fastcall sub_12348F0(__int64 a1, __int64 *a2, __int64 *a3, int a4)
{
  unsigned __int64 v6; // r14
  unsigned int v7; // r12d
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  __int64 v12; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v13[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v14; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 232);
  if ( !(unsigned __int8)sub_122FE20((__int64 **)a1, &v11, a3)
    && !(unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in logical operation") )
  {
    v7 = sub_1224B80((__int64 **)a1, *(_QWORD *)(v11 + 8), &v12, a3);
    if ( !(_BYTE)v7 )
    {
      v9 = *(_QWORD *)(v11 + 8);
      v10 = *(unsigned __int8 *)(v9 + 8);
      if ( (unsigned int)(v10 - 17) <= 1 )
        LOBYTE(v10) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
      if ( (_BYTE)v10 == 12 )
      {
        v14 = 257;
        *a2 = sub_B504D0(a4, v11, v12, (__int64)v13, 0, 0);
        return v7;
      }
      v14 = 259;
      v13[0] = "instruction requires integer or integer vector operands";
      sub_11FD800(a1 + 176, v6, (__int64)v13, 1);
    }
  }
  return 1;
}
