// Function: sub_97DEE0
// Address: 0x97dee0
//
__int64 __fastcall sub_97DEE0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  bool v5; // dl
  __int64 v6; // rcx
  char v7; // si
  bool v8; // al
  __int64 result; // rax

  v4 = *(unsigned int *)(a2 + 32);
  v5 = (unsigned int)(v4 - 24) <= 1 || (((_DWORD)v4 - 31) & 0xFFFFFFFD) == 0;
  if ( (unsigned int)v4 <= 0x1D && (v6 = 537878528, _bittest64(&v6, v4)) )
  {
    v7 = 1;
    if ( (_DWORD)v4 == 13 )
    {
      v8 = 1;
      goto LABEL_5;
    }
  }
  else
  {
    v7 = 0;
  }
  v8 = (_DWORD)v4 == 29 || (_DWORD)v4 == 14;
LABEL_5:
  *(_BYTE *)(a1 + 168) = v5;
  *(_BYTE *)(a1 + 169) = v5;
  *(_BYTE *)(a1 + 170) = v7;
  *(_BYTE *)(a1 + 171) = v8;
  result = (unsigned __int8)sub_CC7F80(a2) == 0 ? 32 : 16;
  *(_DWORD *)(a1 + 172) = result;
  return result;
}
