// Function: sub_16F0D40
// Address: 0x16f0d40
//
__int64 __fastcall sub_16F0D40(unsigned __int64 *a1, unsigned __int64 a2, _QWORD *a3, unsigned __int64 a4, int a5)
{
  unsigned __int64 v7; // rcx
  _BYTE *v9; // rdx
  unsigned int v11; // r13d
  unsigned __int64 v12; // rdi
  unsigned int v13; // eax
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // edi
  unsigned int v18; // r14d
  _BYTE *v19; // r9
  unsigned int v20; // edi

  v7 = *a1;
  v9 = (_BYTE *)*a3;
  if ( *a1 >= a2 )
  {
    v12 = *a1;
    v11 = 0;
  }
  else
  {
    v11 = 0;
    while ( 1 )
    {
      v12 = v7;
      v7 += 4LL;
      v13 = *(_DWORD *)(v7 - 4);
      if ( !a5 && v13 - 55296 <= 0x7FF )
        break;
      if ( v13 <= 0x7F )
      {
        if ( a4 < (unsigned __int64)(v9 + 1) )
          goto LABEL_22;
        *v9++ = v13;
        if ( a2 <= v7 )
        {
LABEL_15:
          v12 = v7;
          goto LABEL_23;
        }
      }
      else
      {
        if ( v13 <= 0x7FF )
        {
          v19 = v9 + 2;
          if ( a4 < (unsigned __int64)(v9 + 2) )
            goto LABEL_22;
          v15 = 1;
          v16 = (v13 >> 6) | 0xFFFFFFC0;
          LOBYTE(v17) = *(_DWORD *)(v7 - 4) & 0x3F | 0x80;
        }
        else
        {
          if ( v13 <= 0xFFFF )
          {
            v14 = v9 + 3;
            if ( a4 < (unsigned __int64)(v9 + 3) )
              goto LABEL_22;
            v15 = 2;
            v16 = (v13 >> 12) | 0xFFFFFFE0;
            LOBYTE(v18) = *(_DWORD *)(v7 - 4) & 0x3F | 0x80;
            v17 = (*(_DWORD *)(v7 - 4) >> 6) & 0x3F | 0xFFFFFF80;
          }
          else if ( v13 <= 0x10FFFF )
          {
            if ( a4 < (unsigned __int64)(v9 + 4) )
            {
LABEL_22:
              v11 = 2;
              goto LABEL_23;
            }
            v14 = v9 + 3;
            v9[3] = v13 & 0x3F | 0x80;
            v20 = v13;
            v18 = (v13 >> 6) & 0x3F | 0xFFFFFF80;
            v16 = (v13 >> 18) | 0xFFFFFFF0;
            v15 = 3;
            v17 = (v20 >> 12) & 0x3F | 0xFFFFFF80;
          }
          else
          {
            v14 = v9 + 3;
            if ( a4 < (unsigned __int64)(v9 + 3) )
              goto LABEL_22;
            v15 = 2;
            LOBYTE(v16) = -17;
            LOBYTE(v17) = -65;
            LOBYTE(v18) = -67;
            v11 = 3;
          }
          *(v14 - 1) = v18;
          v19 = v14 - 1;
        }
        *(v19 - 1) = v17;
        v9 = &v19[v15 - 1];
        *(v19 - 2) = v16;
        if ( a2 <= v7 )
          goto LABEL_15;
      }
    }
    v11 = 3;
  }
LABEL_23:
  *a1 = v12;
  *a3 = v9;
  return v11;
}
