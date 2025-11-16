// Function: sub_6434D0
// Address: 0x6434d0
//
__int64 __fastcall sub_6434D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rsi
  int v3; // ecx
  __int64 i; // r8
  __int64 v5; // rdx
  unsigned __int8 v6; // di
  __int64 j; // rcx
  __int64 k; // rbx
  __int64 v9; // rdx
  __int64 v10; // rdx

  result = *(_QWORD *)(a1 + 168);
  v2 = *(_QWORD *)(result + 152);
  if ( v2 && (*(_BYTE *)(v2 + 29) & 0x20) == 0 )
  {
    result = *(_QWORD *)(v2 + 144);
    v3 = (*(_BYTE *)(a1 + 88) >> 4) & 7;
    for ( i = (unsigned int)(v3 - 2); result; result = *(_QWORD *)(result + 112) )
    {
      *(_BYTE *)(result + 88) = (16 * v3) | *(_BYTE *)(result + 88) & 0x8F;
      if ( (unsigned __int8)(v3 - 2) <= 1u )
        *(_BYTE *)(result + 172) = *(_DWORD *)(result + 160) == 0;
    }
    v5 = *(_QWORD *)(v2 + 112);
    v6 = v3 - 2;
    for ( j = 16 * (v3 & 7u); v5; v5 = *(_QWORD *)(v5 + 112) )
    {
      result = (unsigned int)j | *(_BYTE *)(v5 + 88) & 0x8F;
      *(_BYTE *)(v5 + 88) = j | *(_BYTE *)(v5 + 88) & 0x8F;
      if ( v6 <= 1u )
        *(_BYTE *)(v5 + 136) = 1;
    }
    for ( k = *(_QWORD *)(v2 + 104); k; k = *(_QWORD *)(k + 112) )
    {
      result = *(unsigned __int8 *)(k + 140);
      v9 = (unsigned int)(result - 9);
      if ( (unsigned __int8)(result - 9) <= 2u )
      {
        sub_66A6A0(k, v2, v9, j, i);
        result = sub_6434D0(k);
      }
      else if ( (_BYTE)result == 2 && (*(_BYTE *)(k + 161) & 8) != 0 )
      {
        sub_66A6A0(k, v2, v9, j, i);
        result = *(_QWORD *)(k + 176);
        if ( (*(_BYTE *)result & 1) != 0 )
        {
          v10 = *(_QWORD *)(k + 168);
          if ( (*(_BYTE *)(k + 161) & 0x10) != 0 )
            v10 = *(_QWORD *)(v10 + 96);
          while ( v10 )
          {
            j = *(_BYTE *)(k + 88) & 0x70;
            result = (unsigned int)j | *(_BYTE *)(v10 + 88) & 0x8F;
            *(_BYTE *)(v10 + 88) = j | *(_BYTE *)(v10 + 88) & 0x8F;
            v10 = *(_QWORD *)(v10 + 120);
          }
        }
      }
    }
  }
  return result;
}
