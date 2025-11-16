// Function: sub_1039B70
// Address: 0x1039b70
//
void __fastcall sub_1039B70(_QWORD *a1, __int64 a2, char a3)
{
  unsigned __int8 v3; // al
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax

  *a1 = a2;
  if ( a2 )
  {
    v3 = *(_BYTE *)(a2 - 16);
    if ( a3 )
    {
      if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
      {
        v4 = *(_QWORD *)(a2 - 32);
        v5 = *(unsigned int *)(a2 - 24);
      }
      else
      {
        v5 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
        v4 = a2 - 8LL * ((v3 >> 2) & 0xF) - 16;
      }
      a1[1] = v4 + 8 * v5;
    }
    else
    {
      if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
        v6 = *(_QWORD *)(a2 - 32);
      else
        v6 = a2 - 8LL * ((v3 >> 2) & 0xF) - 16;
      a1[1] = v6;
    }
  }
}
