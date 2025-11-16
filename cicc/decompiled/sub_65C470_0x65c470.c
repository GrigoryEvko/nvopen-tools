// Function: sub_65C470
// Address: 0x65c470
//
__int64 __fastcall sub_65C470(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rcx
  __int64 result; // rax
  __int64 v7; // rbx
  char v8; // dl
  char v9; // al
  char v10; // dl
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 *v16; // rcx
  __int64 v17; // rsi
  char v18; // dl
  __int64 v19; // rsi

  v5 = *(_QWORD *)(a1 + 120);
  result = v5 & 0x108010000000LL;
  if ( (v5 & 0x108010000000LL) != 0x8000000000LL )
    return result;
  v7 = a1;
  if ( (*(_BYTE *)(a1 + 132) & 4) != 0 && (*(_BYTE *)(a1 + 124) & 0x40) != 0 )
    return result;
  result = *(_QWORD *)(a1 + 288);
  if ( unk_4F0774C )
  {
    if ( result )
    {
      v8 = *(_BYTE *)(result + 140);
      if ( v8 != 7 )
      {
        v5 &= 0x40000100000uLL;
        if ( v5 != 0x100000 )
          goto LABEL_10;
        return result;
      }
      if ( (*(_BYTE *)(a1 + 124) & 0x40) != 0 )
        return result;
    }
    result = 0x40000100000LL;
    v5 &= 0x40000100000uLL;
    if ( v5 == 0x100000 )
      return result;
LABEL_12:
    if ( (*(_BYTE *)(a1 + 130) & 0x20) != 0
      || (v9 = *(_BYTE *)(a1 + 124), (v9 & 1) != 0)
      || *(_QWORD *)(a1 + 424) && (*(_WORD *)(a1 + 124) & 0x140) == 0x140 )
    {
      result = *(unsigned __int8 *)(a1 + 125);
      if ( (result & 2) == 0 )
        goto LABEL_32;
    }
    else
    {
      v10 = *(_BYTE *)(a1 + 125);
      if ( (v9 & 0x40) == 0 )
      {
        if ( (v10 & 4) == 0 )
        {
          v17 = a1 + 104;
          a1 = (v10 & 2) == 0 ? 1598 : 2542;
          sub_6851C0(a1, v17);
LABEL_21:
          *(_WORD *)(v7 + 124) &= 0xF87Fu;
          *(_QWORD *)(v7 + 304) = 0;
          result = sub_72C930(a1);
          v11 = *(_QWORD *)v7;
          *(_QWORD *)(v7 + 272) = result;
          *(_QWORD *)(v7 + 280) = result;
          *(_QWORD *)(v7 + 288) = result;
          if ( v11 )
          {
            switch ( *(_BYTE *)(v11 + 80) )
            {
              case 2:
                *(_QWORD *)(*(_QWORD *)(v11 + 88) + 128LL) = result;
                return result;
              case 7:
              case 9:
                v15 = *(_QWORD *)(v11 + 88);
                v16 = (__int64 *)(v15 + 120);
                goto LABEL_42;
              case 8:
                *(_QWORD *)(*(_QWORD *)(v11 + 88) + 120LL) = result;
                return result;
              case 0xA:
              case 0xB:
                *(_QWORD *)(*(_QWORD *)(v11 + 88) + 152LL) = result;
                return result;
              case 0x14:
                return result;
              case 0x15:
                v15 = *(_QWORD *)(*(_QWORD *)(v11 + 88) + 192LL);
                v16 = (__int64 *)(v15 + 120);
LABEL_42:
                *v16 = result;
                if ( v15 )
                  *(_BYTE *)(v15 + 175) &= 0xF8u;
                break;
              default:
                sub_721090(a1);
            }
          }
          return result;
        }
LABEL_20:
        sub_625700(a1);
        goto LABEL_21;
      }
      result = *(_BYTE *)(a1 + 127) & 4;
      if ( (v10 & 4) != 0 )
      {
        if ( !(_BYTE)result )
        {
          result = *(_DWORD *)(a1 + 120) & 0x14000;
          if ( (_DWORD)result != 0x10000 )
            goto LABEL_20;
        }
        if ( (*(_BYTE *)(a1 + 125) & 2) == 0 )
          goto LABEL_33;
      }
      else
      {
        v18 = v10 & 2;
        if ( !(_BYTE)result )
        {
          if ( !*(_QWORD *)a1 || (v19 = a1 + 48, (*(_BYTE *)(*(_QWORD *)a1 + 81LL) & 0x20) != 0) )
            v19 = a1 + 104;
          a1 = v18 == 0 ? 1593 : 2543;
          sub_6851C0(a1, v19);
          goto LABEL_21;
        }
        if ( !v18 )
          return result;
      }
    }
    v12 = 0;
    if ( dword_4F077BC )
      v12 = 32 * (unsigned int)((_DWORD)qword_4F077B4 == 0);
    if ( !(unsigned int)sub_8D97D0(*(_QWORD *)(a1 + 280), *(_QWORD *)(a1 + 304), v12, dword_4F077BC, a5) )
      return sub_6851C0(2541, a1 + 104);
    result = *(unsigned __int8 *)(a1 + 125);
LABEL_32:
    if ( (result & 4) == 0 )
      return result;
LABEL_33:
    v13 = *(_QWORD *)(a1 + 280);
    v14 = *(_QWORD *)(v7 + 304);
    if ( v13 != v14 )
    {
      result = sub_8D97D0(v13, v14, 32, v5, a5);
      if ( !(_DWORD)result )
        return sub_6851C0(2887, v7 + 104);
    }
    return result;
  }
  if ( !result )
    goto LABEL_12;
  while ( 1 )
  {
    v8 = *(_BYTE *)(result + 140);
LABEL_10:
    if ( v8 != 12 )
      break;
    result = *(_QWORD *)(result + 160);
  }
  if ( v8 )
    goto LABEL_12;
  return result;
}
