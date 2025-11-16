// Function: sub_1277B60
// Address: 0x1277b60
//
__int64 __fastcall sub_1277B60(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  int v5; // r8d
  unsigned int v6; // edx
  __int64 *v7; // rbx
  __int64 v8; // rsi

  if ( (*(_BYTE *)(a2 + 144) & 4) != 0 )
    sub_127B550("field number cannot be directly accessed for bitfields!");
  v3 = *(unsigned int *)(a1 + 128);
  v4 = *(_QWORD *)(a1 + 112);
  if ( !(_DWORD)v3 )
    goto LABEL_9;
  v5 = 1;
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    while ( v8 != -8 )
    {
      v6 = (v3 - 1) & (v5 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_4;
      ++v5;
    }
LABEL_9:
    sub_127B550("Unable to look up field information!");
  }
LABEL_4:
  if ( v7 == (__int64 *)(v4 + 16 * v3) )
    goto LABEL_9;
  return *((unsigned int *)v7 + 2);
}
