// Function: sub_23ECEA0
// Address: 0x23ecea0
//
__int64 __fastcall sub_23ECEA0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 *v8; // rdi

  v5 = *(_QWORD *)(a3 + 8);
  v6 = v5;
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
  {
    v6 = **(_QWORD **)(v5 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
      v6 = **(_QWORD **)(v6 + 16);
  }
  if ( *(_DWORD *)(v6 + 8) >> 8 && ((unsigned int)(*(_DWORD *)(a1 + 56) - 26) > 1 || sub_23DC400(v5)) )
    return 1;
  v7 = sub_BD6020(a3);
  if ( (_BYTE)v7 || *(_BYTE *)a3 == 60 && (_BYTE)qword_4FE0C48 && !(unsigned __int8)sub_23ECB50(a1, a3) )
  {
    return 1;
  }
  else
  {
    v8 = *(__int64 **)(a1 + 1024);
    if ( v8 && sub_D904B0(v8, a2) )
      LOBYTE(v7) = sub_98C100(a3, 0) != 0;
  }
  return v7;
}
