// Function: sub_BAED80
// Address: 0xbaed80
//
__int64 __fastcall sub_BAED80(__int64 a1)
{
  __int64 v1; // rcx
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdx
  int v5; // edi
  __int64 *v6; // rcx
  __int64 v7; // rax

  v1 = *(_QWORD *)(a1 + 40);
  LODWORD(v2) = *(_DWORD *)(a1 + 48) - 1;
  if ( (int)v2 < 0 )
  {
    v3 = 0;
LABEL_11:
    v7 = 0;
  }
  else
  {
    v2 = (int)v2;
    v3 = 0;
    while ( 1 )
    {
      v4 = *(_QWORD *)(v1 + 8 * v2);
      v5 = v2;
      if ( (v4 & 4) == 0 )
        break;
      --v2;
      v3 = (unsigned int)(v3 + 1);
      if ( (int)v2 < 0 )
        goto LABEL_11;
    }
    v6 = (__int64 *)(v1 + 8LL * (int)v2);
    v7 = 0;
    while ( (v4 & 2) != 0 )
    {
      --v6;
      if ( v5 == (_DWORD)v7 )
      {
        v7 = (unsigned int)(v5 + 1);
        return (v3 << 32) | v7;
      }
      v4 = *v6;
      v7 = (unsigned int)(v7 + 1);
    }
  }
  return (v3 << 32) | v7;
}
