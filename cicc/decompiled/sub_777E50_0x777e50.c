// Function: sub_777E50
// Address: 0x777e50
//
__int64 __fastcall sub_777E50(__int64 a1, int a2, unsigned __int64 a3, __int64 a4)
{
  char v5; // al
  __int64 v6; // rax
  __int64 result; // rax
  unsigned int v8; // eax
  unsigned int v9; // r8d
  unsigned int v10; // edx
  int v11; // ecx
  unsigned int v12; // esi
  _BYTE *v13; // rax
  int v14; // edi
  unsigned int v15[5]; // [rsp+Ch] [rbp-14h] BYREF

  v5 = *(_BYTE *)(a3 + 140);
  v15[0] = 1;
  if ( (unsigned __int8)(v5 - 8) <= 3u )
  {
    v8 = sub_7764B0(a1, a3, v15);
    v9 = v15[0];
    v10 = v8;
    result = v15[0];
    if ( v15[0] && v10 )
    {
      v11 = (a2 - a4) & 7;
      v12 = ((unsigned int)(a2 - a4) >> 3) + 10;
      do
      {
        while ( 1 )
        {
          v13 = (_BYTE *)(a4 + -v12);
          if ( v11 || v10 <= 7 )
            break;
          v10 -= 8;
          *v13 = 0;
          ++v12;
          if ( !v10 )
            return v9;
        }
        v14 = 1 << v11++;
        *v13 &= ~(_BYTE)v14;
        if ( v11 == 8 )
        {
          ++v12;
          v11 = 0;
        }
        --v10;
      }
      while ( v10 );
      return v9;
    }
  }
  else
  {
    v6 = -(((unsigned int)(a2 - a4) >> 3) + 10);
    *(_BYTE *)(a4 + v6) &= ~(1 << ((a2 - a4) & 7));
    return 1;
  }
  return result;
}
