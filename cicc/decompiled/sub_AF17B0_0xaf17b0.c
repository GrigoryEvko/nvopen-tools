// Function: sub_AF17B0
// Address: 0xaf17b0
//
__int64 __fastcall sub_AF17B0(unsigned int a1, unsigned int a2, unsigned int a3)
{
  unsigned int v3; // r14d
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  unsigned int *v8; // rdi
  int v9; // ecx
  unsigned __int64 v10; // rax
  int v11; // edx
  int v13; // [rsp+0h] [rbp-40h] BYREF
  int v14; // [rsp+4h] [rbp-3Ch] BYREF
  int v15; // [rsp+8h] [rbp-38h] BYREF
  __int64 v16; // [rsp+Ch] [rbp-34h]
  _DWORD v17[11]; // [rsp+14h] [rbp-2Ch] BYREF

  v3 = 0;
  v17[2] = a3;
  v17[0] = a1;
  v17[1] = a2;
  v6 = a1 + a2 + (unsigned __int64)a3;
  if ( v6 )
  {
    v7 = a1;
    v8 = v17;
    v9 = 0;
    v10 = v6 - a1;
    if ( !a1 )
      goto LABEL_7;
    while ( 1 )
    {
      if ( (v7 & 0xFE0) != 0 )
      {
        v3 |= ((2 * ((2 * (_WORD)v7) & 0x1FC0 | v7 & 0x1F)) | 0x40) << v9;
        v11 = 14;
      }
      else
      {
        v3 |= ((2 * (_WORD)v7) & 0x1FFE) << v9;
        v11 = v7 < 0x20 ? 7 : 14;
      }
      v9 += v11;
      ++v8;
      if ( !v10 )
        break;
      while ( 1 )
      {
        v7 = *v8;
        v10 -= *v8;
        if ( *v8 )
          break;
LABEL_7:
        ++v8;
        v3 |= 1 << v9++;
        if ( !v10 )
          goto LABEL_8;
      }
    }
  }
LABEL_8:
  v15 = 0;
  sub_AF16E0(v3, &v13, &v14, &v15);
  if ( v13 == a1 && v14 == a2 && v15 == a3 )
  {
    LODWORD(v16) = v3;
    BYTE4(v16) = 1;
  }
  else
  {
    BYTE4(v16) = 0;
  }
  return v16;
}
