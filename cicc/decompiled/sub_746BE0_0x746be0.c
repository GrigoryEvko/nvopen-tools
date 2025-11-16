// Function: sub_746BE0
// Address: 0x746be0
//
__int64 *__fastcall sub_746BE0(__int64 a1)
{
  __int64 v1; // r13
  char v3; // al
  char v5; // bl
  __int64 v6; // rdi
  __int64 v7; // rdx

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 24LL);
  if ( v1 )
    return (__int64 *)v1;
  v3 = *(_BYTE *)(a1 + 184);
  if ( v3 == 1 )
  {
    v5 = 6;
  }
  else
  {
    v5 = 1;
    if ( v3 != 6 )
      return (__int64 *)v1;
  }
  v6 = *(_QWORD *)(a1 + 48);
  if ( v6 && *(_DWORD *)(v6 + 160) )
    v7 = sub_72B800(v6);
  else
    v7 = qword_4F04C50;
  if ( !v7 )
    return (__int64 *)v1;
  return sub_72DB00(a1, v5, v7);
}
