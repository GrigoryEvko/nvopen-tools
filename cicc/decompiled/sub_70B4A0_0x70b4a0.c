// Function: sub_70B4A0
// Address: 0x70b4a0
//
_BYTE *__fastcall sub_70B4A0(unsigned __int8 a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // rax
  char *v4; // rdx
  int v5; // eax
  __int64 v6; // r15
  char *v7; // r14
  int v8; // edx
  int v10; // [rsp+Ch] [rbp-34h]

  if ( (unsigned __int8)(a1 - 9) > 1u && a1 > 1u )
  {
    if ( a1 != 2 )
    {
      if ( (unsigned __int8)(a1 - 3) <= 1u )
      {
LABEL_23:
        v10 = 0;
        v2 = 8;
        goto LABEL_9;
      }
      if ( a1 != 11 )
      {
        if ( a1 != 12 )
        {
          v2 = n;
          if ( (unsigned __int8)(a1 - 5) > 1u || (v10 = 16 - n, 16 - (int)n <= 0) )
          {
            v10 = 0;
            v5 = 0;
            goto LABEL_8;
          }
LABEL_5:
          v3 = 0;
          do
          {
            v4 = &byte_4F077E0[2 * v3++];
            strcpy(v4, "00");
          }
          while ( v10 > (int)v3 );
          v5 = 2 * v10;
LABEL_8:
          if ( v2 <= 0 )
            goto LABEL_22;
          goto LABEL_9;
        }
        goto LABEL_23;
      }
    }
    v10 = 0;
    v2 = 4;
    goto LABEL_9;
  }
  if ( (unsigned __int8)(a1 - 5) <= 1u )
  {
    v10 = 14;
    v2 = 2;
    goto LABEL_5;
  }
  v10 = 0;
  v2 = 2;
LABEL_9:
  v6 = 0;
  v7 = &byte_4F077E0[2 * v10];
  do
  {
    if ( unk_4F07580 )
      v8 = *(unsigned __int8 *)(a2 + v2 - 1 - (int)v6);
    else
      v8 = *(unsigned __int8 *)(a2 + v6);
    ++v6;
    sprintf(v7, "%02x", v8);
    v7 += 2;
  }
  while ( v2 > (int)v6 );
  v5 = 2 * (v10 + v2);
LABEL_22:
  byte_4F077E0[v5] = 0;
  return byte_4F077E0;
}
