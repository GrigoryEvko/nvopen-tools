// Function: sub_271D830
// Address: 0x271d830
//
__int64 __fastcall sub_271D830(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 result; // rax
  unsigned int v4; // eax
  __int64 v5; // rdx

  sub_271D2C0((_BYTE *)a1);
  v2 = *(_BYTE *)(a1 + 2);
  if ( v2 == 2 )
    return 1;
  if ( v2 <= 2u )
  {
    if ( !v2 )
      return 0;
LABEL_17:
    BUG();
  }
  if ( (unsigned __int8)(v2 - 3) > 2u )
    goto LABEL_17;
  if ( v2 != 3 || (result = 1, *(_QWORD *)(a1 + 16)) )
  {
    ++*(_QWORD *)(a1 + 72);
    if ( *(_BYTE *)(a1 + 100) )
    {
LABEL_11:
      *(_QWORD *)(a1 + 92) = 0;
      return 1;
    }
    v4 = 4 * (*(_DWORD *)(a1 + 92) - *(_DWORD *)(a1 + 96));
    v5 = *(unsigned int *)(a1 + 88);
    if ( v4 < 0x20 )
      v4 = 32;
    if ( (unsigned int)v5 <= v4 )
    {
      memset(*(void **)(a1 + 80), -1, 8 * v5);
      goto LABEL_11;
    }
    sub_C8C990(a1 + 72, a2);
    return 1;
  }
  return result;
}
