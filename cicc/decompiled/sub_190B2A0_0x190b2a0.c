// Function: sub_190B2A0
// Address: 0x190b2a0
//
__int64 __fastcall sub_190B2A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r8
  int v11; // edx
  int v12; // r10d

  if ( !byte_4FAE7A0 || !*(_BYTE *)(a1 + 368) )
  {
    v2 = *(_QWORD *)(a1 + 24);
    v3 = 0;
    v4 = *(unsigned int *)(v2 + 48);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD *)(a2 + 8);
      v6 = *(_QWORD *)(v2 + 32);
      v7 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( v5 == *v8 )
      {
LABEL_5:
        if ( v8 != (__int64 *)(v6 + 16 * v4) )
          return v8[1];
      }
      else
      {
        v11 = 1;
        while ( v9 != -8 )
        {
          v12 = v11 + 1;
          v7 = (v4 - 1) & (v11 + v7);
          v8 = (__int64 *)(v6 + 16LL * v7);
          v9 = *v8;
          if ( v5 == *v8 )
            goto LABEL_5;
          v11 = v12;
        }
      }
      return 0;
    }
    return v3;
  }
  return *(_QWORD *)(a2 + 24);
}
