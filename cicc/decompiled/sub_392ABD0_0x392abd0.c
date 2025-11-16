// Function: sub_392ABD0
// Address: 0x392abd0
//
__int64 __fastcall sub_392ABD0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // rax
  int v4; // ecx
  char v5; // r8
  __int64 v6; // rdi
  _BYTE *v7; // rdx
  unsigned __int8 *v9; // rdx
  __int64 v10; // rdi

  v3 = *(unsigned __int8 **)(a2 + 144);
  v4 = *v3;
  if ( *(v3 - 1) != 46 || (unsigned __int8)(v4 - 48) > 9u )
  {
    v5 = *(_BYTE *)(a2 + 113);
    goto LABEL_3;
  }
  v9 = v3 + 1;
  do
  {
    *(_QWORD *)(a2 + 144) = v9;
    v4 = *v9;
    v3 = v9++;
  }
  while ( (unsigned __int8)(v4 - 48) <= 9u );
  if ( (v4 & 0xDF) == 0x45 )
    goto LABEL_25;
  v5 = *(_BYTE *)(a2 + 113);
  if ( (unsigned __int8)((v4 & 0xDF) - 65) <= 0x19u )
    goto LABEL_3;
  if ( (unsigned __int8)(v4 - 36) > 0x3Bu )
    goto LABEL_25;
  v10 = 0x800000008000401LL;
  if ( _bittest64(&v10, (unsigned int)(v4 - 36)) )
    goto LABEL_3;
  v5 &= (_BYTE)v4 == 64;
  if ( !v5 )
  {
LABEL_25:
    sub_392A800(a1, a2);
    return a1;
  }
  v4 = 64;
LABEL_3:
  v6 = 0x8000000083FF401LL;
  while ( 1 )
  {
    while ( 1 )
    {
      if ( (unsigned __int8)((v4 & 0xDF) - 65) > 0x19u )
      {
        if ( (unsigned __int8)(v4 - 36) > 0x3Bu )
          goto LABEL_6;
        if ( !_bittest64(&v6, (unsigned int)(v4 - 36)) )
          break;
      }
      *(_QWORD *)(a2 + 144) = ++v3;
      v4 = *v3;
    }
    if ( (_BYTE)v4 != 64 || !v5 )
      break;
    *(_QWORD *)(a2 + 144) = ++v3;
    v4 = *v3;
  }
LABEL_6:
  v7 = *(_BYTE **)(a2 + 104);
  if ( v3 == v7 + 1 && *v7 == 46 )
  {
    *(_DWORD *)a1 = 24;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = 1;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v7;
    *(_DWORD *)a1 = 2;
    *(_QWORD *)(a1 + 16) = v3 - v7;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 24) = 0;
  }
  return a1;
}
