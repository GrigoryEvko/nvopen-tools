// Function: sub_1D9BB80
// Address: 0x1d9bb80
//
__int64 __fastcall sub_1D9BB80(_DWORD **a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // edx
  int v4; // esi
  __int64 v5; // rcx
  __int64 v6; // r9
  __int64 v7; // rdi
  __int16 v8; // dx
  _WORD *v9; // rcx
  unsigned __int16 v10; // dx
  _WORD *v11; // r8
  _WORD *v12; // r9
  char v13; // al
  unsigned int v14; // ecx
  unsigned __int16 *v15; // rdi
  int v16; // eax
  unsigned int v17; // esi
  int v18; // esi

  result = 0;
  if ( !*(_BYTE *)a2 )
  {
    v3 = *(_DWORD *)(a2 + 8);
    if ( v3 )
    {
      if ( (*(_BYTE *)(a2 + 3) & 0x10) != 0 )
      {
        v4 = *a1[1];
        if ( v4 != v3 )
        {
          if ( v3 < 0 || v4 < 0 )
            return 0;
          v5 = *((_QWORD *)*a1 + 30);
          v6 = *(_QWORD *)(v5 + 8);
          v7 = *(_QWORD *)(v5 + 56);
          LODWORD(v5) = *(_DWORD *)(v6 + 24LL * (unsigned int)v3 + 16);
          v8 = (v5 & 0xF) * v3;
          v9 = (_WORD *)(v7 + 2LL * ((unsigned int)v5 >> 4));
          v10 = *v9 + v8;
          v11 = v9 + 1;
          LODWORD(v9) = *(_DWORD *)(v6 + 24LL * (unsigned int)v4 + 16);
          v12 = (_WORD *)(v7 + 2LL * ((unsigned int)v9 >> 4));
          v13 = (char)v9;
          v14 = v10;
          v15 = v12 + 1;
          v16 = v4 * (v13 & 0xF);
          LOWORD(v16) = *v12 + v16;
          v17 = (unsigned __int16)v16;
          while ( v14 != v17 )
          {
            if ( v14 >= v17 )
            {
              v18 = *v15;
              if ( !(_WORD)v18 )
                return 0;
              v16 += v18;
              ++v15;
              v17 = (unsigned __int16)v16;
            }
            else
            {
              v10 += *v11;
              if ( !*v11 )
                return 0;
              ++v11;
              v14 = v10;
            }
          }
        }
        return 1;
      }
    }
  }
  return result;
}
