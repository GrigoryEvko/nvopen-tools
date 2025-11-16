// Function: sub_325E1D0
// Address: 0x325e1d0
//
__int64 __fastcall sub_325E1D0(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        _BYTE *a4,
        _BYTE *a5,
        __int64 a6,
        __int64 a7)
{
  int v7; // eax
  unsigned int v8; // r10d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax

  v7 = *(_DWORD *)(a1 + 24);
  v8 = 0;
  switch ( v7 )
  {
    case 298:
      if ( (*(_WORD *)(a1 + 32) & 0x380) == 0 )
      {
        v9 = *(unsigned __int16 *)(a1 + 96);
        if ( (_WORD)v9 )
        {
          if ( (((int)*(unsigned __int16 *)(a7 + 2 * (a2 + 5 * v9 + 259392) + 10) >> 4) & 0xB) == 0
            || (((int)*(unsigned __int16 *)(a7 + 2 * (5 * v9 + a3 + 259392) + 10) >> 4) & 0xB) == 0 )
          {
            v10 = *(_QWORD *)(a1 + 40);
            v8 = 1;
            *(_QWORD *)a6 = *(_QWORD *)(v10 + 40);
            *(_DWORD *)(a6 + 8) = *(_DWORD *)(v10 + 48);
          }
        }
      }
      break;
    case 299:
      if ( (*(_WORD *)(a1 + 32) & 0x380) == 0 )
      {
        v12 = *(unsigned __int16 *)(a1 + 96);
        if ( (_WORD)v12 )
        {
          v13 = 5 * v12;
          if ( (*(_BYTE *)(a7 + 2 * (a2 + v13 + 259392) + 10) & 0xB) == 0
            || (*(_BYTE *)(a7 + 2 * (v13 + a3 + 259392) + 10) & 0xB) == 0 )
          {
            v14 = *(_QWORD *)(a1 + 40);
            v8 = 1;
            *(_QWORD *)a6 = *(_QWORD *)(v14 + 80);
            *(_DWORD *)(a6 + 8) = *(_DWORD *)(v14 + 88);
            *a4 = 0;
          }
        }
      }
      break;
    case 362:
      if ( (*(_WORD *)(a1 + 32) & 0x380) == 0 )
      {
        v15 = *(unsigned __int16 *)(a1 + 96);
        if ( (_WORD)v15 )
        {
          v16 = 5 * v15;
          v17 = (int)*(unsigned __int16 *)(a7 + 2 * (a2 + 5 * v15 + 259392) + 10) >> 12;
          if ( !v17
            || v17 == 4
            || (LOBYTE(v8) = (int)*(unsigned __int16 *)(a7 + 2 * (v16 + a3 + 259392) + 10) >> 12 == 4
                          || (int)*(unsigned __int16 *)(a7 + 2 * (v16 + a3 + 259392) + 10) >> 12 == 0,
                (_BYTE)v8) )
          {
            v18 = *(_QWORD *)(a1 + 40);
            v8 = 1;
            *(_QWORD *)a6 = *(_QWORD *)(v18 + 40);
            *(_DWORD *)(a6 + 8) = *(_DWORD *)(v18 + 48);
            *a5 = 1;
          }
        }
      }
      break;
    default:
      if ( v7 == 363 && (*(_WORD *)(a1 + 32) & 0x380) == 0 )
      {
        v19 = *(unsigned __int16 *)(a1 + 96);
        if ( (_WORD)v19 )
        {
          v20 = 5 * v19;
          if ( (*(_BYTE *)(a7 + 2 * (a2 + v20 + 259392) + 11) & 0xB) == 0
            || (*(_BYTE *)(a7 + 2 * (v20 + a3 + 259392) + 11) & 0xB) == 0 )
          {
            v21 = *(_QWORD *)(a1 + 40);
            v8 = 1;
            *(_QWORD *)a6 = *(_QWORD *)(v21 + 80);
            *(_DWORD *)(a6 + 8) = *(_DWORD *)(v21 + 88);
            *a4 = 0;
            *a5 = 1;
          }
        }
      }
      break;
  }
  return v8;
}
