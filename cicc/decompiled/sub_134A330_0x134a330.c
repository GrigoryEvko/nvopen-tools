// Function: sub_134A330
// Address: 0x134a330
//
__int64 __fastcall sub_134A330(_QWORD *a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  unsigned __int64 v5; // rcx
  unsigned int v6; // r12d
  unsigned __int64 v8; // r10
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  bool v11; // zf
  unsigned __int64 v12; // r10
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rax

  v5 = a2[10];
  v6 = 0;
  if ( v5 == 512 )
    return v6;
  v8 = v5 >> 6;
  v9 = a2[(v5 >> 6) + 2] & (-1LL << (v5 & 0x3F));
  if ( !v9 )
  {
    v10 = v8 + 1;
    if ( v8 == 7 )
      return v6;
    while ( 1 )
    {
      v9 = a2[v10 + 2];
      v8 = v10;
      if ( v9 )
        break;
      if ( ++v10 == 8 )
        return 0;
    }
  }
  v11 = !_BitScanForward64(&v9, v9);
  if ( v11 )
    LODWORD(v9) = -1;
  v12 = (int)v9 + (v8 << 6);
  if ( v12 != 512 )
  {
    v13 = v12 >> 6;
    v14 = ~a2[(v12 >> 6) + 2] & (-1LL << (v12 & 0x3F));
    if ( !v14 )
    {
      v15 = v13 + 1;
      if ( v13 == 7 )
      {
LABEL_20:
        v16 = 512;
        v17 = 512;
LABEL_18:
        v18 = v17 - v12;
        v6 = 1;
        *a3 = *a1 + (v12 << 12);
        *a4 = v18 << 12;
        *a2 += v18;
        a2[10] = v16;
        return v6;
      }
      while ( 1 )
      {
        v13 = v15;
        v14 = ~a2[v15 + 2];
        if ( a2[v15 + 2] != -1 )
          break;
        if ( ++v15 == 8 )
          goto LABEL_20;
      }
    }
    v11 = !_BitScanForward64(&v14, v14);
    if ( v11 )
      LODWORD(v14) = -1;
    v16 = (int)v14 + (v13 << 6);
    v17 = v16;
    if ( v16 <= 0x200 )
      goto LABEL_18;
    goto LABEL_20;
  }
  return 0;
}
