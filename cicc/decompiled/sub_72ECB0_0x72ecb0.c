// Function: sub_72ECB0
// Address: 0x72ecb0
//
__int64 *__fastcall sub_72ECB0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v4; // rdi
  __int64 v5; // rdx
  char v6; // si
  char v7; // al
  __int64 v8; // rax

  v1 = *(_QWORD *)(a1 + 144);
  if ( v1 )
    return (__int64 *)v1;
  if ( *(_BYTE *)(a1 + 173) != 12 )
  {
    if ( (*(_BYTE *)(a1 + 172) & 4) != 0 )
      goto LABEL_5;
    return (__int64 *)v1;
  }
  v7 = *(_BYTE *)(a1 + 176);
  if ( v7 == 1 )
  {
    v8 = *(_QWORD *)(a1 + 184);
  }
  else
  {
    if ( (unsigned __int8)(v7 - 5) > 5u )
      goto LABEL_17;
    v8 = *(_QWORD *)(a1 + 192);
  }
  if ( v8 )
    return (__int64 *)v8;
LABEL_17:
  if ( (*(_BYTE *)(a1 + 172) & 4) == 0 && (*(_BYTE *)(a1 + 177) & 0x10) == 0 )
    return (__int64 *)v1;
LABEL_5:
  v4 = *(_QWORD *)(a1 + 48);
  if ( v4 && *(_DWORD *)(v4 + 160) )
    v5 = sub_72B800(v4);
  else
    v5 = qword_4F04C50;
  if ( !v5 )
    return (__int64 *)v1;
  v6 = 8;
  if ( (*(_BYTE *)(a1 + 172) & 4) == 0 )
    v6 = 3;
  return sub_72DB00(a1, v6, v5);
}
