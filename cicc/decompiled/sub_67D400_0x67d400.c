// Function: sub_67D400
// Address: 0x67d400
//
__int64 __fastcall sub_67D400(int *a1, unsigned __int8 a2, _DWORD *a3)
{
  unsigned __int8 v4; // al
  char v6[12]; // [rsp+Ch] [rbp-14h] BYREF

  v6[0] = a2;
  if ( a2 <= 7u )
  {
    sub_67C4B0(a1, v6, a3);
    v4 = v6[0];
    if ( v6[0] <= 7u )
    {
      if ( v6[0] != 7 )
        return unk_4F07481 <= v4;
      if ( (unsigned int)sub_729F80((unsigned int)*a3) )
      {
        v4 = v6[0];
        return unk_4F07481 <= v4;
      }
    }
  }
  return 1;
}
