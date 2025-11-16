// Function: sub_25DDE50
// Address: 0x25dde50
//
char __fastcall sub_25DDE50(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char result; // al
  __int64 v4; // [rsp+0h] [rbp-30h]

  v2 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  result = sub_B19DB0(*(_QWORD *)a1, a2, **(_QWORD **)(a1 + 8));
  if ( result )
  {
    v4 = sub_9208B0(*(_QWORD *)(a1 + 16), **(_QWORD **)(a1 + 24));
    return (unsigned __int64)(v4 + 7) >> 3 <= (unsigned __int64)(sub_9208B0(*(_QWORD *)(a1 + 16), v2) + 7) >> 3;
  }
  return result;
}
