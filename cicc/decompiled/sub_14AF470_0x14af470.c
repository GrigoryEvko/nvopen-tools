// Function: sub_14AF470
// Address: 0x14af470
//
__int64 __fastcall sub_14AF470(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v4; // r12d
  unsigned __int8 v5; // dl
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int16 v12; // ax
  int v14; // eax
  __int64 v15; // rax
  _QWORD **v16; // r14
  unsigned int v17; // ebx
  _QWORD *v18; // rax
  _BYTE *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned int v22; // r8d
  __int64 v23; // r8
  _QWORD **v24; // r13
  unsigned int v25; // ebx
  _QWORD *v26; // rax
  __int64 v27; // rax
  unsigned int v28; // ecx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 *v35; // [rsp+18h] [rbp-48h] BYREF
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF
  _QWORD *v37[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *(_BYTE *)(a1 + 16);
  LOBYTE(v4) = v5 > 0x17u || v5 == 5;
  if ( (_BYTE)v4 )
  {
    if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
    {
      v8 = 0;
      v9 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      do
      {
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v10 = *(_QWORD *)(a1 - 8);
        else
          v10 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v11 = *(_QWORD *)(v10 + v8);
        if ( *(_BYTE *)(v11 + 16) <= 0x10u && (unsigned __int8)sub_1593DF0(v11, a2) )
          return 0;
        v8 += 24;
      }
      while ( v8 != v9 );
      v5 = *(_BYTE *)(a1 + 16);
    }
    if ( v5 <= 0x17u )
      v14 = *(unsigned __int16 *)(a1 + 18);
    else
      v14 = v5 - 24;
    switch ( v14 )
    {
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 29:
      case 31:
      case 33:
      case 34:
      case 35:
      case 49:
      case 50:
      case 53:
      case 58:
      case 64:
        return 0;
      case 17:
      case 20:
        v37[0] = &v36;
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v23 = *(_QWORD *)(a1 - 8);
        else
          v23 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v4 = sub_13D2630(v37, *(_BYTE **)(v23 + 24));
        if ( !(_BYTE)v4 )
          return v4;
        v24 = (_QWORD **)v36;
        v25 = *(_DWORD *)(v36 + 8);
        if ( v25 <= 0x40 )
        {
          v26 = *(_QWORD **)v36;
LABEL_30:
          LOBYTE(v4) = v26 != 0;
          return v4;
        }
        if ( v25 - (unsigned int)sub_16A57B0(v36) <= 0x40 )
        {
          v26 = (_QWORD *)**v24;
          goto LABEL_30;
        }
        return v4;
      case 18:
      case 21:
        v37[0] = &v36;
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
          v15 = *(_QWORD *)(a1 - 8);
        else
          v15 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v4 = sub_13D2630(v37, *(_BYTE **)(v15 + 24));
        if ( !(_BYTE)v4 )
          return 0;
        v16 = (_QWORD **)v36;
        v17 = *(_DWORD *)(v36 + 8);
        if ( v17 <= 0x40 )
        {
          v18 = *(_QWORD **)v36;
          if ( !*(_QWORD *)v36 )
            return 0;
LABEL_22:
          if ( v18 != (_QWORD *)-1LL )
            return v4;
          v37[0] = &v35;
          v19 = *(_BYTE **)sub_13CF970(a1);
          if ( (unsigned __int8)sub_13D2630(v37, v19) )
            return (unsigned int)sub_13CFF40(v35, (__int64)v19, v20, v21, v22) ^ 1;
          return 0;
        }
        if ( v17 - (unsigned int)sub_16A57B0(v36) > 0x40 )
          return v4;
        v18 = (_QWORD *)**v16;
        if ( v18 )
          goto LABEL_22;
        break;
      case 30:
        v12 = *(_WORD *)(a1 + 18);
        if ( ((v12 >> 7) & 6) != 0 )
          return 0;
        if ( (v12 & 1) != 0 )
          return 0;
        v29 = sub_15F2060(a1);
        if ( (unsigned __int8)sub_1560180(v29 + 112, 45) )
          return 0;
        v30 = sub_15F2060(a1);
        if ( (unsigned __int8)sub_1560180(v30 + 112, 42) )
          return 0;
        v31 = sub_15F2060(a1);
        if ( (unsigned __int8)sub_1560180(v31 + 112, 43) )
          return 0;
        v32 = sub_15F2050(a1);
        v33 = sub_1632FA0(v32);
        return sub_13F8190(*(_QWORD *)(a1 - 24), 1 << (*(unsigned __int16 *)(a1 + 18) >> 1) >> 1, v33, a2, a3);
      case 54:
        if ( !(_BYTE)a4 )
          return sub_1C307B0(a1);
        if ( v5 != 78 )
          return 0;
        v27 = *(_QWORD *)(a1 - 24);
        if ( *(_BYTE *)(v27 + 16) || (*(_BYTE *)(v27 + 33) & 0x20) == 0 )
          return 0;
        v28 = *(_DWORD *)(v27 + 36);
        if ( v28 > 0xD3 )
        {
          if ( v28 <= 0x1072 )
          {
            v4 = 0;
            if ( v28 >= 0x1071 )
              return a4;
          }
          else
          {
            switch ( v28 )
            {
              case 0x10BEu:
              case 0x10BFu:
              case 0x10C0u:
              case 0x10E9u:
              case 0x10EAu:
              case 0x10EBu:
              case 0x10EEu:
              case 0x10EFu:
              case 0x10F0u:
              case 0x10F8u:
              case 0x10F9u:
              case 0x10FAu:
              case 0x10FCu:
                v4 = a4;
                break;
              default:
                return 0;
            }
          }
        }
        else if ( v28 > 0xBC )
        {
          v4 = 0;
          if ( ((1LL << ((unsigned __int8)v28 + 67)) & 0x700241) != 0 )
            return a4;
        }
        else if ( v28 > 0x26 )
        {
          v4 = 0;
          if ( v28 == 144 )
            return a4;
        }
        else
        {
          if ( v28 <= 5 )
            return 0;
          v4 = 0;
          if ( ((1LL << v28) & 0x5380000040LL) != 0 )
            return a4;
        }
        return v4;
      default:
        return v4;
    }
  }
  return 0;
}
