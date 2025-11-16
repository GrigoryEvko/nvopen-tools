// Function: sub_7D9E50
// Address: 0x7d9e50
//
void __fastcall sub_7D9E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  _QWORD *v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  v6 = *(_BYTE *)(a1 + 57);
  if ( v6 <= 4u )
  {
    if ( v6 > 2u )
    {
      v7 = *(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL);
      if ( (unsigned int)sub_8D2B20(*v7) )
        sub_7D9DD0(v7, a2, v8, v9, v10, v11);
    }
  }
  else if ( v6 == 5 )
  {
    sub_7D9DD0(*(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL), a2, a3, a4, a5, a6);
    sub_7D8240((__int64 *)a1);
  }
}
