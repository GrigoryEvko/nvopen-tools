// Function: sub_85B070
// Address: 0x85b070
//
__int64 __fastcall sub_85B070(int a1)
{
  __int64 v1; // rdi
  __int64 v2; // r9
  __int64 v3; // rax
  int v4; // r8d
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 result; // rax

  v1 = 776LL * a1;
  v2 = qword_4F04C68[0] + v1;
  v3 = v1 - 776;
  v4 = dword_4F04C64;
  v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(qword_4F04C68[0] + v1 - 772) == 8 )
  {
    do
    {
      v6 = v3;
      v3 -= 776;
    }
    while ( *(_BYTE *)(qword_4F04C68[0] + v3 + 4) == 8 );
    v7 = v6 + qword_4F04C68[0];
  }
  else
  {
    v7 = qword_4F04C68[0] + v1;
  }
  if ( v2 == v5 - 776 )
  {
    *(_DWORD *)(v5 + 552) = *(_DWORD *)(v7 + 552);
    *(_DWORD *)(v7 + 552) = v4;
    goto LABEL_6;
  }
  *(_DWORD *)(v5 + 552) = dword_4F04C64 - 1;
  result = 0xFFFFFFFFLL;
  *(_DWORD *)(v7 + 552) = v4;
  if ( v2 )
LABEL_6:
    result = 1594008481 * (unsigned int)(v1 >> 3);
  dword_4F04C60 = result;
  return result;
}
