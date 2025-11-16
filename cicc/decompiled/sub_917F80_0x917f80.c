// Function: sub_917F80
// Address: 0x917f80
//
__int64 __fastcall sub_917F80(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // edx
  __int64 *v6; // rbx
  __int64 v7; // rsi
  int v9; // r8d

  if ( (*(_BYTE *)(a2 + 144) & 4) != 0 )
    sub_91B8A0("field number cannot be directly accessed for bitfields!");
  v3 = *(unsigned int *)(a1 + 128);
  v4 = *(_QWORD *)(a1 + 112);
  if ( !(_DWORD)v3 )
    goto LABEL_7;
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v4 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v9 = 1;
    while ( v7 != -4096 )
    {
      v5 = (v3 - 1) & (v9 + v5);
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_4;
      ++v9;
    }
LABEL_7:
    sub_91B8A0("Unable to look up field information!");
  }
LABEL_4:
  if ( v6 == (__int64 *)(v4 + 16 * v3) )
    goto LABEL_7;
  return *((unsigned int *)v6 + 2);
}
