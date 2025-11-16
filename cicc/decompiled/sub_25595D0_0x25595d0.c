// Function: sub_25595D0
// Address: 0x25595d0
//
__int64 __fastcall sub_25595D0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int16 v4; // bx
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_WORD *)(a1 + 98);
  result = v4 & 3;
  if ( (v4 & 3) == 3 )
  {
    result = (unsigned int)((char)sub_2509800((_QWORD *)(a1 + 72)) - 6);
    if ( (unsigned int)result <= 1 )
    {
      if ( (v4 & 7) == 7 )
      {
        v16[0] = sub_A778C0(a3, 89, 0);
        return sub_25594F0(a4, v16, v12, v13, v14, v15);
      }
      else if ( (_BYTE)qword_4FEF9E8 )
      {
        v16[0] = sub_A78730(a3, "no-capture-maybe-returned", 0x19u, 0, 0);
        return sub_25594F0(a4, v16, v8, v9, v10, v11);
      }
    }
  }
  return result;
}
