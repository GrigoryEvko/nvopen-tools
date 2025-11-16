// Function: sub_217E810
// Address: 0x217e810
//
__int64 __fastcall sub_217E810(__int64 a1, int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rax
  __int64 v9; // r12
  __int16 *v10; // rsi
  unsigned __int16 v11; // dx
  unsigned int v12; // edi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v16; // rax
  void *v17; // rax
  __int64 *v18; // r13
  __int64 *v19[8]; // [rsp+0h] [rbp-40h] BYREF

  v8 = sub_217D7E0(a2, *(_QWORD *)(a1 + 248), 0, a4, a5, a6);
  v9 = v8;
  if ( !v8 )
  {
    if ( byte_4FD2E80 )
    {
      v17 = sub_16E8CB0();
      sub_1263B40((__int64)v17, "\tMULTIDEF\n");
      return v9;
    }
    return 0;
  }
  if ( sub_1E17880(v8) )
    return 0;
  v10 = *(__int16 **)(v9 + 16);
  if ( (*((_QWORD *)v10 + 2) & 0x3F80LL) != 0 )
    return 0;
  v11 = *v10;
  if ( (unsigned __int16)*v10 > 0x188u )
  {
    if ( v11 > 0x2BCu )
    {
      if ( v11 > 0xB70u )
      {
        if ( v11 > 0x114Du )
        {
          if ( (unsigned __int16)(v11 - 4442) <= 5u )
            return 0;
        }
        else
        {
          if ( v11 > 0x1147u )
            return 0;
          if ( v11 <= 0xCD7u )
          {
            if ( v11 > 0xCD1u )
              return 0;
          }
          else if ( (unsigned __int16)(v11 - 3449) <= 5u )
          {
            return 0;
          }
        }
      }
      else
      {
        if ( v11 > 0xB61u || v11 == 776 )
          return 0;
        if ( v11 <= 0x308u )
        {
          if ( v11 <= 0x2D7u )
          {
            if ( v11 > 0x2D4u )
              return 0;
          }
          else if ( (unsigned __int16)(v11 - 759) <= 0xEu )
          {
            return 0;
          }
        }
        else if ( (unsigned __int16)(v11 - 817) <= 0xFu )
        {
          return 0;
        }
      }
    }
    else
    {
      if ( v11 > 0x2B9u )
        return 0;
      if ( v11 > 0x261u )
      {
        if ( (unsigned __int16)(v11 - 680) <= 1u )
          return 0;
      }
      else
      {
        if ( v11 > 0x215u )
        {
          switch ( v11 )
          {
            case 0x216u:
            case 0x217u:
            case 0x218u:
            case 0x219u:
            case 0x21Au:
            case 0x21Bu:
            case 0x21Cu:
            case 0x21Du:
            case 0x21Eu:
            case 0x21Fu:
            case 0x224u:
            case 0x225u:
            case 0x226u:
            case 0x227u:
            case 0x228u:
            case 0x229u:
            case 0x22Au:
            case 0x22Bu:
            case 0x22Cu:
            case 0x22Du:
            case 0x232u:
            case 0x233u:
            case 0x234u:
            case 0x235u:
            case 0x236u:
            case 0x237u:
            case 0x238u:
            case 0x239u:
            case 0x23Au:
            case 0x23Bu:
            case 0x244u:
            case 0x245u:
            case 0x246u:
            case 0x247u:
            case 0x248u:
            case 0x24Bu:
            case 0x24Cu:
            case 0x24Du:
            case 0x24Eu:
            case 0x24Fu:
            case 0x250u:
            case 0x251u:
            case 0x254u:
            case 0x255u:
            case 0x256u:
            case 0x257u:
            case 0x258u:
            case 0x259u:
            case 0x25Au:
            case 0x25Du:
            case 0x25Eu:
            case 0x25Fu:
            case 0x260u:
            case 0x261u:
              return 0;
            default:
              goto LABEL_8;
          }
        }
        if ( v11 > 0x1B0u )
        {
          if ( (unsigned __int16)(v11 - 435) <= 0xCu )
            return 0;
        }
        else if ( v11 > 0x1AAu )
        {
          return 0;
        }
      }
    }
  }
  else if ( v11 > 0x17Cu )
  {
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 280) + 8LL) == 62 )
      return 0;
  }
  else if ( v11 == 165 || (unsigned __int16)(v11 - 174) <= 4u )
  {
    return 0;
  }
LABEL_8:
  v12 = *(_DWORD *)(v9 + 40);
  if ( v12 )
  {
    v13 = *(_QWORD *)(v9 + 32);
    v14 = v13 + 40LL * (v12 - 1) + 40;
    while ( *(_BYTE *)v13 || (*(_BYTE *)(v13 + 3) & 0x10) == 0 || a2 == *(_DWORD *)(v13 + 8) )
    {
      v13 += 40;
      if ( v14 == v13 )
      {
        if ( (v10[5] & 1) == 0 )
          goto LABEL_15;
        if ( v12 > 2 )
        {
          v16 = *(_QWORD *)(v9 + 32);
          if ( *(_BYTE *)(v16 + 80) == 1 && (((unsigned int)*(_QWORD *)(v16 + 104) - 2) & 0xFFFFFFFD) == 0 )
            return v9;
        }
        return 0;
      }
    }
    return 0;
  }
  if ( (v10[5] & 1) == 0 )
  {
LABEL_15:
    if ( !v11 || v11 == 45 || (v11 & 0xFFF7) == 1 )
      return 0;
    if ( v11 > 0xC32u )
    {
      if ( v11 == 3258 )
        return 0;
    }
    else
    {
      if ( v11 <= 0xC15u )
      {
        if ( (unsigned __int16)(v11 - 328) > 0x1Fu || ((1LL << ((unsigned __int8)v11 - 72)) & 0xFFE7FDFF) == 0 )
          goto LABEL_64;
        return 0;
      }
      if ( ((1LL << ((unsigned __int8)v11 - 22)) & 0x1E7E7E3F) != 0 )
        return 0;
    }
LABEL_64:
    v18 = (__int64 *)(*(_QWORD *)(a1 + 328) + 8LL * *(unsigned int *)(a1 + 344));
    sub_217E750(v19, (__int64 *)(a1 + 320), v9);
    if ( v18 != v19[2] )
      return 0;
    return v9;
  }
  return 0;
}
