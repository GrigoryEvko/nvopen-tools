// Function: sub_1580C80
// Address: 0x1580c80
//
__int64 __fastcall sub_1580C80(__int64 a1)
{
  char v1; // al
  __int64 result; // rax
  int v3; // eax
  _QWORD *v4; // rbx
  __int64 v5; // r12

  v1 = *(_BYTE *)(a1 + 8);
  if ( v1 == 13 )
  {
LABEL_2:
    result = (*(_DWORD *)(a1 + 8) & 0x100) == 0;
    if ( (*(_DWORD *)(a1 + 8) & 0x100) != 0 )
    {
      v3 = *(_DWORD *)(a1 + 12);
      if ( v3 )
      {
        v4 = *(_QWORD **)(a1 + 16);
        v5 = (__int64)&v4[(unsigned int)(v3 - 1) + 1];
        do
        {
          result = sub_1580C80(*v4);
          if ( !(_BYTE)result )
            break;
          ++v4;
        }
        while ( v4 != (_QWORD *)v5 );
      }
      else
      {
        return 1;
      }
    }
  }
  else
  {
    while ( v1 == 14 )
    {
      a1 = *(_QWORD *)(a1 + 24);
      v1 = *(_BYTE *)(a1 + 8);
      if ( v1 == 13 )
        goto LABEL_2;
    }
    return 0;
  }
  return result;
}
