// Function: sub_1B2A9B0
// Address: 0x1b2a9b0
//
void __fastcall sub_1B2A9B0(char *src, char *a2, __int64 a3)
{
  __int64 v3; // r12
  char *v4; // r14
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 *i; // r15
  __int64 v8; // rdx

  if ( src != a2 && a2 != src + 8 )
  {
    v3 = a3 + 40;
    v4 = src + 8;
    while ( 1 )
    {
      while ( sub_1B29A30(v3, *(_QWORD *)v4, *(_QWORD *)src) )
      {
        v5 = *(_QWORD *)v4;
        if ( src != v4 )
          memmove(src + 8, src, v4 - src);
        *(_QWORD *)src = v5;
        v4 += 8;
        if ( a2 == v4 )
          return;
      }
      v6 = *(_QWORD *)v4;
      for ( i = (__int64 *)v4; ; --i )
      {
        v8 = *(i - 1);
        if ( !v6 || *(_BYTE *)(v6 + 16) != 17 )
          break;
        if ( v8 && *(_BYTE *)(v8 + 16) == 17 && *(_DWORD *)(v6 + 32) >= *(_DWORD *)(v8 + 32) )
          goto LABEL_17;
LABEL_12:
        *i = v8;
      }
      if ( (!v8 || *(_BYTE *)(v8 + 16) != 17) && sub_1B298A0(v3, v6, v8) )
        break;
LABEL_17:
      *i = v6;
      v4 += 8;
      if ( a2 == v4 )
        return;
    }
    v8 = *(i - 1);
    goto LABEL_12;
  }
}
