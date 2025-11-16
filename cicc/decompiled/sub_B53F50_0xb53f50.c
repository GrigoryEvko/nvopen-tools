// Function: sub_B53F50
// Address: 0xb53f50
//
__int64 __fastcall sub_B53F50(__int64 a1)
{
  __int64 v1; // rbp
  _DWORD *v3; // rax
  __int64 v4; // rcx
  _DWORD *v5; // rdi
  __int64 v6; // rsi
  _DWORD *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r8
  _QWORD v13[4]; // [rsp-20h] [rbp-20h] BYREF

  if ( *(_BYTE *)(a1 + 56) )
  {
    v13[3] = v1;
    v3 = *(_DWORD **)(a1 + 8);
    v4 = *(unsigned int *)(a1 + 16);
    v5 = &v3[v4];
    v6 = (4 * v4) >> 2;
    if ( (4 * v4) >> 4 )
    {
      v7 = &v3[4 * ((4 * v4) >> 4)];
      while ( !*v3 )
      {
        if ( v3[1] )
        {
          ++v3;
          goto LABEL_9;
        }
        if ( v3[2] )
        {
          v3 += 2;
          goto LABEL_9;
        }
        if ( v3[3] )
        {
          v3 += 3;
          goto LABEL_9;
        }
        v3 += 4;
        if ( v3 == v7 )
        {
          v6 = v5 - v3;
          goto LABEL_16;
        }
      }
      goto LABEL_9;
    }
LABEL_16:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return 0;
LABEL_19:
        v11 = 0;
        if ( !*v3 )
          return v11;
LABEL_9:
        if ( (unsigned int)v4 > 1 && v5 != v3 )
        {
          v8 = sub_AA48A0(*(_QWORD *)(*(_QWORD *)a1 + 40LL));
          v9 = *(_QWORD *)(a1 + 8);
          v10 = *(unsigned int *)(a1 + 16);
          v13[0] = v8;
          return sub_B8C150(v13, v9, v10, 0);
        }
        return 0;
      }
      if ( *v3 )
        goto LABEL_9;
      ++v3;
    }
    if ( *v3 )
      goto LABEL_9;
    ++v3;
    goto LABEL_19;
  }
  return 0;
}
