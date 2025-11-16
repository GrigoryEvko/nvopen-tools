// Function: sub_F6C0B0
// Address: 0xf6c0b0
//
__int64 __fastcall sub_F6C0B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r14
  unsigned __int64 v3; // rdx
  int v4; // r12d
  unsigned int i; // r15d
  __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx

  v1 = sub_D47930(a1);
  v2 = v1;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 == v1 + 48 )
      goto LABEL_21;
    if ( !v3 )
      BUG();
    v2 = v3 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
LABEL_21:
      BUG();
    if ( *(_BYTE *)(v3 - 24) == 31 && (*(_DWORD *)(v3 - 20) & 0x7FFFFFF) == 3 )
    {
      v4 = sub_B46E30(v3 - 24);
      if ( v4 )
      {
        for ( i = 0; i != v4; ++i )
        {
          while ( 1 )
          {
            v6 = sub_B46EC0(v2, i);
            if ( !*(_BYTE *)(a1 + 84) )
              break;
            v7 = *(_QWORD **)(a1 + 64);
            v8 = &v7[*(unsigned int *)(a1 + 76)];
            if ( v7 == v8 )
              return v2;
            while ( v6 != *v7 )
            {
              if ( v8 == ++v7 )
                return v2;
            }
            if ( v4 == ++i )
              return 0;
          }
          if ( !sub_C8CA60(a1 + 56, v6) )
            return v2;
        }
      }
    }
    return 0;
  }
  return v2;
}
