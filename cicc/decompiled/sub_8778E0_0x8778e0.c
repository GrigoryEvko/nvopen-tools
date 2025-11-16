// Function: sub_8778E0
// Address: 0x8778e0
//
__int64 *__fastcall sub_8778E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // al
  __int64 **v7; // rdx

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 40LL);
  v3 = **(_QWORD **)(*(_QWORD *)(a2 + 56) + 168LL);
  if ( !v3 )
    return 0;
  while ( 1 )
  {
    v5 = *(_QWORD *)(v3 + 40);
    if ( v2 == v5 || v2 && v5 && dword_4F07588 && (v4 = *(_QWORD *)(v5 + 32), *(_QWORD *)(v2 + 32) == v4) && v4 )
    {
      v6 = *(_BYTE *)(v3 + 96);
      if ( (v6 & 4) == 0 )
        goto LABEL_13;
      if ( (unsigned int)sub_8D5D50(v3, a2) )
        break;
    }
    v3 = *(_QWORD *)v3;
    if ( !v3 )
      return 0;
  }
  v6 = *(_BYTE *)(v3 + 96);
LABEL_13:
  if ( (v6 & 2) != 0 )
    v7 = sub_72B780(v3);
  else
    v7 = *(__int64 ***)(v3 + 112);
  return v7[1];
}
