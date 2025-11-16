// Function: sub_7325D0
// Address: 0x7325d0
//
__int64 __fastcall sub_7325D0(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 i; // r12
  __int64 v4; // rbx
  __int64 j; // r13
  unsigned int v6; // [rsp-2Ch] [rbp-2Ch] BYREF

  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v4 = *(_QWORD *)(i + 168);
    if ( (*(_BYTE *)(v4 + 16) & 0x20) == 0 )
    {
      for ( j = *(_QWORD *)(i + 160); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      result = sub_732490(j, &v6);
      if ( (_DWORD)result )
      {
        *(_BYTE *)(v4 + 16) |= 0x20u;
        if ( (*(_BYTE *)(j + 176) & 0x20) != 0 )
        {
          if ( *a2 )
          {
            result = (__int64)&dword_4D041AC;
            if ( !dword_4D041AC )
            {
              result = (__int64)&dword_4F077BC;
              if ( dword_4F077BC )
              {
                if ( !(_DWORD)qword_4F077B4 )
                {
                  result = (__int64)&qword_4F077A8;
                  if ( qword_4F077A8 > 0xC34Fu )
                    return sub_5EB950(8u, 323, j, (__int64)a2);
                }
              }
              else if ( !(_DWORD)qword_4F077B4 )
              {
                return sub_5EB950(8u, 323, j, (__int64)a2);
              }
            }
          }
        }
      }
      else
      {
        result = v6;
        if ( v6 )
          return sub_880320(j, 1, i, 6, a2);
      }
    }
  }
  return result;
}
