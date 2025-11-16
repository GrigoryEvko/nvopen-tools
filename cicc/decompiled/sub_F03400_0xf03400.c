// Function: sub_F03400
// Address: 0xf03400
//
__int64 __fastcall sub_F03400(unsigned __int64 *a1, unsigned __int64 a2, __int64 *a3, unsigned __int64 a4, int a5)
{
  unsigned __int16 *v8; // rdx
  __int64 v9; // rcx
  int v11; // edi
  unsigned int v12; // eax
  unsigned __int16 *v13; // rdi
  char v14; // dl
  __int64 v15; // r14
  unsigned int v16; // eax
  char v17; // si
  __int64 v18; // rcx
  char v19; // di
  __int64 v20; // r13
  unsigned __int16 *v21; // r9
  __int64 v22; // rdi

  v8 = (unsigned __int16 *)*a1;
  v9 = *a3;
  if ( *a1 >= a2 )
  {
LABEL_15:
    *a1 = (unsigned __int64)v8;
    *a3 = v9;
    return 0;
  }
  while ( 1 )
  {
    v16 = *v8;
    v21 = v8 + 1;
    if ( v16 - 55296 <= 0x3FF )
    {
      if ( a2 <= (unsigned __int64)v21 )
      {
        *a1 = (unsigned __int64)v8;
        *a3 = v9;
        return 1;
      }
      v11 = v8[1];
      if ( (unsigned int)(v11 - 56320) > 0x3FF )
      {
        if ( !a5 )
          break;
LABEL_17:
        v15 = v9 + 3;
        if ( a4 < v9 + 3 )
          goto LABEL_19;
        ++v8;
        v18 = 2;
        v17 = -32;
      }
      else
      {
        v12 = v11 + ((v16 - 55296) << 10) + 9216;
        v13 = v8 + 2;
        if ( a4 < v9 + 4 )
          goto LABEL_19;
        v14 = v12;
        v15 = v9 + 3;
        v16 = v12 >> 6;
        v17 = -16;
        *(_BYTE *)(v9 + 3) = v14 & 0x3F | 0x80;
        v8 = v13;
        v18 = 3;
      }
      v19 = v16;
      v20 = v15 - 1;
      v16 >>= 6;
      *(_BYTE *)(v15 - 1) = v19 & 0x3F | 0x80;
      goto LABEL_8;
    }
    if ( !a5 && v16 - 56320 <= 0x3FF )
      break;
    if ( v16 > 0x7F )
    {
      if ( v16 > 0x7FF )
        goto LABEL_17;
      v20 = v9 + 2;
      if ( a4 < v9 + 2 )
      {
LABEL_19:
        *a1 = (unsigned __int64)v8;
        *a3 = v9;
        return 2;
      }
      ++v8;
      v18 = 1;
      v17 = -64;
LABEL_8:
      v9 = v20 - 1 + v18;
      *(_BYTE *)(v20 - 1) = v16 & 0x3F | 0x80;
      *(_BYTE *)(v20 - 2) = v17 | (v16 >> 6);
      if ( a2 <= (unsigned __int64)v8 )
        goto LABEL_15;
    }
    else
    {
      v22 = v9 + 1;
      if ( v9 + 1 > a4 )
        goto LABEL_19;
      ++v8;
      ++v9;
      *(_BYTE *)(v22 - 1) = v16;
      if ( a2 <= (unsigned __int64)v21 )
        goto LABEL_15;
    }
  }
  *a1 = (unsigned __int64)v8;
  *a3 = v9;
  return 3;
}
