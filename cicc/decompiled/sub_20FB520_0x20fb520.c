// Function: sub_20FB520
// Address: 0x20fb520
//
__int64 __fastcall sub_20FB520(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax

  if ( a2 )
  {
    v4 = 0;
    v5 = *(unsigned int *)(a2 + 8);
    if ( (_DWORD)v5 == 2 )
      v4 = *(_QWORD *)(a2 - 8);
    v6 = sub_20FB390(a1, *(unsigned __int8 **)(a2 - 8 * v5), v4);
    v7 = v6;
    if ( v6 )
    {
      if ( a1[28] == v6 && *a1 == *(_QWORD *)(a3 + 56) )
        return 1;
      v8 = *(_QWORD *)(a3 + 32);
      v9 = a3 + 24;
      if ( v8 != v9 )
      {
        while ( 1 )
        {
          v10 = sub_15C70A0(v8 + 64);
          if ( v10 )
          {
            v11 = *(unsigned int *)(v10 + 8);
            v12 = 0;
            if ( (_DWORD)v11 == 2 )
              v12 = *(_QWORD *)(v10 - 8);
            v13 = sub_20FB390(a1, *(unsigned __int8 **)(v10 - 8 * v11), v12);
            if ( v13 )
            {
              if ( v7 == v13
                || *(_DWORD *)(v7 + 176) < *(_DWORD *)(v13 + 176) && *(_DWORD *)(v7 + 180) > *(_DWORD *)(v13 + 180) )
              {
                break;
              }
            }
          }
          if ( !v8 )
            BUG();
          if ( (*(_BYTE *)v8 & 4) != 0 )
          {
            v8 = *(_QWORD *)(v8 + 8);
            if ( v9 == v8 )
              return 0;
          }
          else
          {
            while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
              v8 = *(_QWORD *)(v8 + 8);
            v8 = *(_QWORD *)(v8 + 8);
            if ( v9 == v8 )
              return 0;
          }
        }
        return 1;
      }
    }
    return 0;
  }
  return 0;
}
