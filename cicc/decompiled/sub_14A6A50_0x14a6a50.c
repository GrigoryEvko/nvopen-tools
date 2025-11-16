// Function: sub_14A6A50
// Address: 0x14a6a50
//
__int64 __fastcall sub_14A6A50(__int64 a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 v4; // r13
  int v5; // eax
  unsigned int v6; // ecx
  int v7; // r15d
  int v8; // edx
  int v9; // eax
  __int64 v10; // rdi
  _BYTE **v12; // [rsp+0h] [rbp-40h]
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v3 = *(_DWORD *)(a1 + 8);
  if ( v3 <= 2 )
  {
    v5 = 1;
    v6 = 2;
  }
  else
  {
    v4 = v3;
    if ( (unsigned __int8)(**(_BYTE **)(a1 - 8LL * v3) - 4) > 0x1Eu )
    {
      v13 = (v3 - 1) >> 1;
      goto LABEL_5;
    }
    v5 = 3;
    v6 = 3;
  }
  v4 = v3;
  v13 = (v3 - v5) / v6;
  if ( v3 - v5 < v6 )
    return 0;
LABEL_5:
  v7 = 0;
  v12 = (_BYTE **)(a1 - 8 * v4);
  while ( 1 )
  {
    if ( v3 <= 2 )
    {
      v8 = 1;
      v9 = 2;
    }
    else
    {
      v8 = (unsigned __int8)(**v12 - 4) < 0x1Fu ? 3 : 1;
      v9 = 3 - ((unsigned __int8)(**v12 - 4) >= 0x1Fu);
    }
    v10 = *(_QWORD *)(a1 + 8 * ((unsigned int)(v8 + v7 * v9) - v4));
    if ( a2 == v10 || (unsigned __int8)sub_14A6A50(v10, a2) )
      break;
    if ( ++v7 == v13 )
      return 0;
  }
  return 1;
}
