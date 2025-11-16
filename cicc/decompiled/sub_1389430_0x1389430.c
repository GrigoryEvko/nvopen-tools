// Function: sub_1389430
// Address: 0x1389430
//
__int64 __fastcall sub_1389430(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rax

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 > 3u )
  {
    if ( v3 == 5 )
    {
      result = (unsigned int)*(unsigned __int16 *)(a2 + 18) - 51;
      if ( (unsigned int)result > 1 )
      {
        result = sub_13848E0(*(_QWORD *)(a1 + 24), a2, 0, 0);
        if ( (_BYTE)result )
          return (__int64)sub_1389140(a1, a2);
      }
    }
    else
    {
      return sub_13848E0(*(_QWORD *)(a1 + 24), a2, 0, a3);
    }
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 24);
    v5 = sub_14C81A0(a2);
    result = sub_13848E0(v4, a2, 0, v5);
    if ( (_BYTE)result )
    {
      v7 = *(_QWORD *)(a1 + 24);
      v8 = sub_14C8160();
      return sub_13848E0(v7, a2, 1u, v8);
    }
  }
  return result;
}
