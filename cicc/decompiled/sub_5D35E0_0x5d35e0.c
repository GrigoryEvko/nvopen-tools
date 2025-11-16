// Function: sub_5D35E0
// Address: 0x5d35e0
//
__int64 __fastcall sub_5D35E0(__int64 a1)
{
  __int64 v1; // rbx
  char i; // al
  __int64 v3; // r12
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rax
  __int64 v8; // rdx

  v1 = *(_QWORD *)(a1 + 120);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  if ( (*(_BYTE *)(a1 + 144) & 4) != 0 )
  {
    v5 = *(_QWORD *)(a1 + 176);
    if ( unk_4F068C0 && (v6 = *(unsigned __int8 *)(a1 + 137), v6 >= v5) && ((_BYTE)v6 || *(_QWORD *)(a1 + 112)) )
    {
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 140LL) == 11 )
        return *(_QWORD *)(v1 + 128);
      else
        return *(_QWORD *)(qword_4CF7C90 + 128) + qword_4CF7C98;
    }
    else
    {
      return *(_QWORD *)(a1 + 128) + (*(unsigned __int8 *)(a1 + 136) + v5 + dword_4F06BA0 - 1) / dword_4F06BA0;
    }
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 128);
    if ( (*(_BYTE *)(a1 + 146) & 4) != 0 && (unsigned __int8)(i - 9) <= 1u )
    {
      v7 = sub_730E80(v1);
      v8 = *(_QWORD *)(v1 + 168);
      if ( *(_QWORD *)(v8 + 32) >= v7 )
        v7 = *(_QWORD *)(v8 + 32);
      return v3 + v7;
    }
    else
    {
      return v3 + *(_QWORD *)(v1 + 128);
    }
  }
}
