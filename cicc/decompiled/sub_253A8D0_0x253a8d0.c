// Function: sub_253A8D0
// Address: 0x253a8d0
//
bool __fastcall sub_253A8D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r8
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r10
  __int64 v8; // rax
  int v9; // edx
  int v10; // r11d

  if ( !*(_BYTE *)(a1 + 97) )
    return 0;
  v3 = *(unsigned int *)(a1 + 128);
  v4 = *(_QWORD *)(a1 + 112);
  if ( !(_DWORD)v3 )
    return 0;
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v4 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = (v3 - 1) & (v9 + v5);
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_5;
      v9 = v10;
    }
    return 0;
  }
LABEL_5:
  if ( v6 == (__int64 *)(v4 + 16 * v3) )
    return 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 136) + 16LL * *((unsigned int *)v6 + 2) + 8);
  if ( !v8 )
    return 0;
  return *(_DWORD *)(v8 + 12) != 2;
}
