// Function: sub_2AC38D0
// Address: 0x2ac38d0
//
__int64 __fastcall sub_2AC38D0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v7; // rax

  v3 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    v4 = 0;
  }
  else
  {
    if ( v3 == 61 )
    {
      v4 = *(_QWORD *)(a2 - 32);
      v5 = *(_QWORD *)(a2 + 8);
      goto LABEL_7;
    }
    v4 = 0;
    if ( v3 == 62 )
      v4 = *(_QWORD *)(a2 - 32);
  }
  v5 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
LABEL_7:
  if ( !(unsigned int)sub_31A5150(*(_QWORD *)(a1 + 440), v5, v4) || sub_2AC3650(a1, (unsigned __int8 *)a2, a3) )
    return 0;
  v7 = sub_B43CC0(a2);
  return (unsigned int)sub_2AAE050(v5, v7) ^ 1;
}
