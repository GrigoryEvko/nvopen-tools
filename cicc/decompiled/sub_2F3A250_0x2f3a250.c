// Function: sub_2F3A250
// Address: 0x2f3a250
//
void __fastcall sub_2F3A250(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r12
  bool v4; // zf
  _BYTE *v5; // rsi
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a2 + 120);
  for ( i = v2 + 16LL * *(unsigned int *)(a2 + 128); v2 != i; v2 += 16 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)v2 ^ 6LL;
      v7 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
      v8[0] = v7;
      if ( (v6 & 6) != 0 || *(_DWORD *)(v2 + 8) <= 3u )
        break;
      v2 += 16;
      --*(_DWORD *)(v7 + 224);
      if ( v2 == i )
        return;
    }
    v4 = (*(_DWORD *)(v7 + 216))-- == 1;
    if ( v4 && v7 != a1 + 328 )
    {
      v5 = *(_BYTE **)(a1 + 3544);
      if ( v5 == *(_BYTE **)(a1 + 3552) )
      {
        sub_2ECAD30(a1 + 3536, v5, v8);
      }
      else
      {
        if ( v5 )
        {
          *(_QWORD *)v5 = v7;
          v5 = *(_BYTE **)(a1 + 3544);
        }
        *(_QWORD *)(a1 + 3544) = v5 + 8;
      }
    }
  }
}
