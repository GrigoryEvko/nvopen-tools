// Function: sub_2255110
// Address: 0x2255110
//
__int64 __fastcall sub_2255110(__int64 a1, _BYTE *a2, char a3)
{
  __int64 result; // rax
  _WORD *v4; // rcx
  int v5; // edi
  _BYTE *v6; // rsi
  _BYTE *v7; // rdx
  int v8; // ecx

  result = *(unsigned int *)(a1 + 24);
  v4 = a2 + 1;
  *a2 = 37;
  if ( (result & 0x800) != 0 )
  {
    a2[1] = 43;
    v4 = a2 + 2;
  }
  if ( (result & 0x400) != 0 )
  {
    *(_BYTE *)v4 = 35;
    v4 = (_WORD *)((char *)v4 + 1);
  }
  v5 = result & 0x104;
  if ( v5 == 260 )
  {
    v6 = v4;
    if ( !a3 )
      goto LABEL_16;
LABEL_12:
    *v6++ = a3;
    v7 = v6 + 1;
    if ( v5 != 4 )
      goto LABEL_8;
LABEL_13:
    *v6 = 102;
    *v7 = 0;
    return result;
  }
  v6 = v4 + 1;
  *v4 = 10798;
  if ( a3 )
    goto LABEL_12;
  v7 = (char *)v4 + 3;
  if ( v5 == 4 )
    goto LABEL_13;
LABEL_8:
  v8 = result & 0x4000;
  if ( v5 == 256 )
  {
    *v6 = v8 == 0 ? 101 : 69;
    *v7 = 0;
    return v8 == 0 ? 101 : 69;
  }
  if ( v5 != 260 )
  {
    *v6 = v8 == 0 ? 103 : 71;
    *v7 = 0;
    return v8 == 0 ? 103 : 71;
  }
  v4 = v6;
LABEL_16:
  result = (result & 0x4000) == 0 ? 97 : 65;
  *v4 = (unsigned __int8)result;
  return result;
}
