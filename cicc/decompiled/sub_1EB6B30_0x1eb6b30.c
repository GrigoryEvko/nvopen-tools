// Function: sub_1EB6B30
// Address: 0x1eb6b30
//
__int64 __fastcall sub_1EB6B30(__int64 a1, int a2, unsigned __int16 a3)
{
  unsigned int v4; // esi
  __int64 v5; // rdx
  __int64 v7; // rdi
  __int64 v8; // r9
  unsigned int v9; // eax
  __int64 v10; // r8
  int v11; // edx

  v4 = a2 & 0x7FFFFFFF;
  v5 = *(_QWORD *)(a1 + 600);
  v7 = *(unsigned int *)(a1 + 400);
  v8 = *(_QWORD *)(a1 + 392);
  v9 = *(unsigned __int8 *)(v5 + v4);
  if ( v9 < (unsigned int)v7 )
  {
    while ( 1 )
    {
      v10 = v8 + 24LL * v9;
      v11 = *(_DWORD *)(v10 + 8);
      if ( v4 == (v11 & 0x7FFFFFFF) )
        break;
      v9 += 256;
      if ( (unsigned int)v7 <= v9 )
        goto LABEL_6;
    }
  }
  else
  {
LABEL_6:
    v10 = v8 + 24 * v7;
    v11 = *(_DWORD *)(v10 + 8);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 648) + 4LL * a3) = v11;
  *(_WORD *)(v10 + 12) = a3;
  return v10;
}
