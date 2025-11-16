// Function: sub_BC8CE0
// Address: 0xbc8ce0
//
_BOOL8 __fastcall sub_BC8CE0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12
  unsigned __int8 v3; // al
  __int64 *v4; // rdx
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  _WORD *v8; // rax
  __int64 v9; // rdx
  _BOOL4 v10; // r8d
  unsigned int i; // edx
  __int64 v13; // rcx
  __int64 v14; // rcx
  _QWORD *v15; // rax
  unsigned __int8 v16; // al
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax

  *a2 = 0;
  if ( a1 )
  {
    v2 = a1 - 16;
    v3 = *(_BYTE *)(a1 - 16);
    if ( (v3 & 2) != 0 )
      v4 = *(__int64 **)(a1 - 32);
    else
      v4 = (__int64 *)(v2 - 8LL * ((v3 >> 2) & 0xF));
    v5 = *v4;
    if ( !*(_BYTE *)*v4 )
    {
      v6 = sub_B91420(*v4);
      if ( v7 == 14
        && *(_QWORD *)v6 == 0x775F68636E617262LL
        && *(_DWORD *)(v6 + 8) == 1751607653
        && *(_WORD *)(v6 + 12) == 29556 )
      {
        for ( i = sub_BC8810(a1); ; ++i )
        {
          v10 = (*(_BYTE *)(a1 - 16) & 2) != 0;
          if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
          {
            if ( i >= *(_DWORD *)(a1 - 24) )
              return v10;
            v13 = *(_QWORD *)(a1 - 32);
          }
          else
          {
            if ( i >= ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) )
              return 1;
            v13 = v2 - 8LL * ((*(_BYTE *)(a1 - 16) >> 2) & 0xF);
          }
          v14 = *(_QWORD *)(*(_QWORD *)(v13 + 8LL * i) + 136LL);
          v15 = *(_QWORD **)(v14 + 24);
          if ( *(_DWORD *)(v14 + 32) > 0x40u )
            v15 = (_QWORD *)*v15;
          *a2 += v15;
        }
      }
      v8 = (_WORD *)sub_B91420(v5);
      if ( v9 == 2 && *v8 == 20566 )
      {
        v16 = *(_BYTE *)(a1 - 16);
        if ( (v16 & 2) != 0 )
        {
          if ( *(_DWORD *)(a1 - 24) > 3u )
          {
            v17 = *(_QWORD *)(a1 - 32);
LABEL_29:
            v18 = *(_QWORD *)(v17 + 16);
            if ( *(_BYTE *)v18 != 1 || (v19 = *(_QWORD *)(v18 + 136), *(_BYTE *)v19 != 17) )
              BUG();
            v20 = *(_QWORD **)(v19 + 24);
            if ( *(_DWORD *)(v19 + 32) > 0x40u )
              v20 = (_QWORD *)*v20;
            *a2 = v20;
            return 1;
          }
        }
        else if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xFu) > 3 )
        {
          v17 = v2 - 8LL * ((v16 >> 2) & 0xF);
          goto LABEL_29;
        }
      }
    }
    return 0;
  }
  return 0;
}
