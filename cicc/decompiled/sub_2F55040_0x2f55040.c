// Function: sub_2F55040
// Address: 0x2f55040
//
__int64 __fastcall sub_2F55040(__int64 a1)
{
  __int64 v1; // rdx
  int v2; // r14d
  int v3; // r12d
  __int64 v4; // rsi
  __int64 v5; // rax
  unsigned int v7[9]; // [rsp+Ch] [rbp-24h] BYREF

  v1 = *(_QWORD *)(a1 + 16);
  v2 = *(_DWORD *)(v1 + 64);
  if ( v2 )
  {
    v3 = 0;
    while ( 1 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(v1 + 56) + 16LL * (v3 & 0x7FFFFFFF) + 8);
      if ( v5 )
      {
        if ( (*(_BYTE *)(v5 + 4) & 8) != 0 )
        {
          while ( 1 )
          {
            v5 = *(_QWORD *)(v5 + 32);
            if ( !v5 )
              break;
            if ( (*(_BYTE *)(v5 + 4) & 8) == 0 )
            {
              if ( !*(_QWORD *)(a1 + 384) )
                return 1;
              goto LABEL_4;
            }
          }
          if ( v2 != ++v3 )
            goto LABEL_6;
          return 0;
        }
        if ( !*(_QWORD *)(a1 + 384) )
          return 1;
LABEL_4:
        v4 = *(_QWORD *)(a1 + 8);
        v7[0] = v3 | 0x80000000;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, unsigned int *))(a1 + 392))(
               a1 + 368,
               v4,
               v1,
               v7) )
        {
          return 1;
        }
      }
      if ( v2 == ++v3 )
        return 0;
LABEL_6:
      v1 = *(_QWORD *)(a1 + 16);
    }
  }
  return 0;
}
