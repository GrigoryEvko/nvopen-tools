// Function: sub_71BD50
// Address: 0x71bd50
//
void __fastcall sub_71BD50(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  int v3; // eax

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  if ( (*(_BYTE *)(v1 + 177) & 1) == 0 )
    return;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) > 1u )
  {
    if ( (*(_BYTE *)(a1 + 192) & 2) == 0 || (*(_BYTE *)(a1 + 195) & 0x40) != 0 )
      return;
    v2 = sub_735B60(v1, 0);
    if ( v2 )
    {
      if ( a1 == v2 )
      {
LABEL_13:
        v1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
        goto LABEL_14;
      }
      if ( (*(_BYTE *)(v2 + 193) & 0x20) == 0 && !*(_DWORD *)(v2 + 160) && !*(_QWORD *)(v2 + 344) )
        return;
      v3 = (*(_BYTE *)(v2 + 195) & 0x40) == 0;
    }
    else
    {
      v3 = *(_BYTE *)(a1 + 206) >> 7;
    }
    if ( !v3 )
      return;
    goto LABEL_13;
  }
LABEL_14:
  sub_71BC30(v1);
  if ( *(char *)(a1 + 206) < 0 && (*(_BYTE *)(v1 + 141) & 0x40) != 0 )
    sub_8DCD80(v1);
}
