// Function: sub_2539E30
// Address: 0x2539e30
//
__int64 __fastcall sub_2539E30(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned int v4; // r8d
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rax

  v2 = 0x186000000000001LL;
  while ( 1 )
  {
    v4 = sub_B46490(a1);
    if ( (_BYTE)v4 )
    {
      if ( *(_BYTE *)a1 != 85 )
        return v4;
      v6 = *(_QWORD *)(a1 - 32);
      if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(a1 + 80) || (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
        return v4;
      v7 = *(_DWORD *)(v6 + 36);
      if ( v7 <= 0xD3 )
      {
        if ( v7 > 0x9A )
        {
          if ( !_bittest64(&v2, v7 - 155) )
            return v4;
        }
        else if ( v7 != 11 && v7 - 68 > 3 )
        {
          return v4;
        }
        goto LABEL_13;
      }
      if ( v7 != 324 )
      {
        if ( v7 > 0x144 )
        {
          if ( v7 != 376 )
            return v4;
          goto LABEL_13;
        }
        if ( v7 != 282 && v7 - 291 > 1 )
          return v4;
      }
    }
LABEL_13:
    v8 = sub_B46B10(a1, 0);
    a1 = v8;
    if ( !v8 || v8 == a2 )
      return 0;
  }
}
