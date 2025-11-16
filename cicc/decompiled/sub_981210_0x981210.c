// Function: sub_981210
// Address: 0x981210
//
bool __fastcall sub_981210(__int64 a1, __int64 a2, unsigned int *a3)
{
  unsigned int v4; // edx
  __int64 *v5; // r14
  _BYTE *v7; // rax
  size_t v8; // rdx

  if ( (*(_BYTE *)(a2 + 33) & 0x20) == 0 )
  {
    v4 = *(_DWORD *)(a2 + 132);
    v5 = *(__int64 **)(a2 + 40);
    if ( v4 == -1 )
    {
      v7 = (_BYTE *)sub_BD5D20(a2);
      if ( !(unsigned __int8)sub_980AF0(a1, v7, v8, (_DWORD *)(a2 + 132)) )
      {
        *(_DWORD *)(a2 + 132) = 524;
        return 0;
      }
      v4 = *(_DWORD *)(a2 + 132);
    }
    if ( v4 != 524 )
    {
      *a3 = v4;
      return sub_97FAA0(a1, *(_QWORD *)(a2 + 24), v4, v5);
    }
    return 0;
  }
  return 0;
}
