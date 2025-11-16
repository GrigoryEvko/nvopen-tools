// Function: sub_39DFB50
// Address: 0x39dfb50
//
void __fastcall sub_39DFB50(__int64 a1, __int64 a2, char a3)
{
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rax

  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
  {
    sub_16E2F40(a2, a1 + 448);
    if ( a3 )
    {
      v6 = *(unsigned int *)(a1 + 456);
      if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 460) )
      {
        sub_16CD150(a1 + 448, (const void *)(a1 + 464), 0, 1, v4, v5);
        v6 = *(unsigned int *)(a1 + 456);
      }
      *(_BYTE *)(*(_QWORD *)(a1 + 448) + v6) = 10;
      ++*(_DWORD *)(a1 + 456);
    }
  }
}
