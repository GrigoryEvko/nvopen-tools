// Function: sub_2F54D60
// Address: 0x2f54d60
//
unsigned __int64 __fastcall sub_2F54D60(__int64 a1)
{
  __int64 (*v1)(void); // rdx
  unsigned __int64 result; // rax
  unsigned int v3[3]; // [rsp+Ch] [rbp-14h] BYREF

  v1 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 8) + 472LL);
  LODWORD(result) = qword_5023DE8;
  if ( v1 != sub_2F4C070 )
  {
    LODWORD(result) = v1();
    if ( (unsigned int)qword_5023DE8 >= (unsigned int)result )
      LODWORD(result) = qword_5023DE8;
  }
  result = (unsigned int)result;
  *(_QWORD *)(a1 + 28944) = (unsigned int)result;
  if ( (_DWORD)result )
  {
    result = sub_2E3A080(*(_QWORD *)(a1 + 792));
    if ( result )
    {
      if ( result <= 0x3FFF )
      {
        sub_F02DB0(v3, result, 0x4000u);
        return (unsigned __int64)sub_1098CF0((unsigned __int64 *)(a1 + 28944), v3[0]);
      }
      else if ( result <= 0xFFFFFFFF )
      {
        sub_F02DB0(v3, 0x4000u, result);
        return (unsigned __int64)sub_1098D40((unsigned __int64 *)(a1 + 28944), v3[0]);
      }
      else
      {
        result = *(_QWORD *)(a1 + 28944) * (result >> 14);
        *(_QWORD *)(a1 + 28944) = result;
      }
    }
    else
    {
      *(_QWORD *)(a1 + 28944) = 0;
    }
  }
  return result;
}
