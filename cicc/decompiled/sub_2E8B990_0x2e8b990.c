// Function: sub_2E8B990
// Address: 0x2e8b990
//
__int64 __fastcall sub_2E8B990(__int64 a1)
{
  unsigned __int16 *v1; // r14
  __int64 v2; // rbx
  unsigned int v3; // r15d
  __int64 v4; // rax
  __int16 v5; // dx
  unsigned __int16 v6; // ax
  int v7; // r13d

  v1 = *(unsigned __int16 **)(a1 + 16);
  if ( *v1 == 32 )
    return 1;
  v2 = 0;
  v3 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v3 )
  {
    do
    {
      v4 = *(_QWORD *)(a1 + 32) + 40 * v2;
      if ( !*(_BYTE *)v4 && (*(_BYTE *)(v4 + 3) & 0x10) == 0 )
      {
        v5 = *(_WORD *)(v4 + 2) & 0xFF0;
        if ( v1[1] > (unsigned int)v2 )
        {
          v6 = v1[20 * *v1 + 22 + 3 * v1[8] + 3 * v2];
          if ( (v6 & 1) != 0 )
          {
            v7 = (unsigned __int8)(v6 >> 4);
            if ( !v5 )
              return 1;
LABEL_8:
            if ( (unsigned int)sub_2E89F40(a1, v2) != v7 )
              return 1;
            goto LABEL_9;
          }
        }
        if ( v5 )
        {
          v7 = -1;
          goto LABEL_8;
        }
      }
LABEL_9:
      ++v2;
    }
    while ( v3 > (unsigned int)v2 );
  }
  return 0;
}
