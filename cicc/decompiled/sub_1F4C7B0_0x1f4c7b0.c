// Function: sub_1F4C7B0
// Address: 0x1f4c7b0
//
__int64 __fastcall sub_1F4C7B0(int a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  __int64 v5; // rcx
  __int64 v7; // rdx

  if ( a1 < 0 )
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 16LL * (a1 & 0x7FFFFFFF) + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a3 + 272) + 8LL * (unsigned int)a1);
  v4 = 0;
  if ( v3 )
  {
    if ( (*(_BYTE *)(v3 + 3) & 0x10) != 0 || (v3 = *(_QWORD *)(v3 + 32)) != 0 && (*(_BYTE *)(v3 + 3) & 0x10) != 0 )
    {
      v5 = *(_QWORD *)(v3 + 16);
      v4 = 0;
      if ( a2 == *(_QWORD *)(v5 + 24) )
        goto LABEL_11;
      while ( 1 )
      {
        v3 = *(_QWORD *)(v3 + 32);
        if ( !v3 || (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
          break;
        v7 = *(_QWORD *)(v3 + 16);
        if ( v5 != v7 )
        {
          v5 = *(_QWORD *)(v3 + 16);
          if ( a2 == *(_QWORD *)(v7 + 24) )
          {
LABEL_11:
            if ( **(_WORD **)(v5 + 16) != 12 )
            {
              if ( v4 )
              {
                if ( v5 != v4 )
                  return 0;
              }
              else
              {
                v4 = v5;
              }
            }
          }
        }
      }
    }
  }
  return v4;
}
