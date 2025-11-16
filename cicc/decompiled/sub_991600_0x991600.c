// Function: sub_991600
// Address: 0x991600
//
char __fastcall sub_991600(int a1, __int64 a2, int a3, int a4, __int64 a5, __int64 a6, __int64 a7, __int64 a8)
{
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  _BYTE *v13; // r13
  unsigned int v14; // ebx
  __int64 *v15; // r12
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned int v19; // r8d
  __int64 v20; // r12
  __int64 v21; // rdi
  _BYTE *v22; // rbx
  unsigned int v23; // r12d
  int v24; // r10d
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v27; // ecx
  _BYTE *v28; // rax
  _BYTE *v29; // rax
  int v30; // r8d
  __int64 *v34; // [rsp+18h] [rbp-38h] BYREF
  __int64 **v35; // [rsp+20h] [rbp-30h] BYREF
  char v36; // [rsp+28h] [rbp-28h]

  switch ( a1 )
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
    case 11:
    case 31:
    case 33:
    case 35:
    case 36:
    case 37:
    case 51:
    case 52:
    case 55:
    case 60:
    case 66:
      goto LABEL_4;
    case 19:
    case 22:
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v20 = *(_QWORD *)(a2 - 8);
      else
        v20 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v21 = *(_QWORD *)(v20 + 32);
      v22 = (_BYTE *)(v21 + 24);
      if ( *(_BYTE *)v21 == 17 )
        goto LABEL_21;
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v21 + 8) + 8LL) - 17 > 1 )
        goto LABEL_4;
      if ( *(_BYTE *)v21 > 0x15u )
        goto LABEL_4;
      v29 = (_BYTE *)sub_AD7630(v21, 0);
      if ( !v29 || *v29 != 17 )
        goto LABEL_4;
      v22 = v29 + 24;
LABEL_21:
      v23 = *((_DWORD *)v22 + 2);
      if ( v23 <= 0x40 )
      {
        v10 = *(_QWORD **)v22;
      }
      else
      {
        if ( v23 - (unsigned int)sub_C444A0(v22) > 0x40 )
          goto LABEL_7;
        v10 = **(_QWORD ***)v22;
      }
      LOBYTE(v10) = v10 != 0;
      return (char)v10;
    case 20:
    case 23:
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v11 = *(_QWORD *)(a2 - 8);
      else
        v11 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v12 = *(_QWORD *)(v11 + 32);
      v13 = (_BYTE *)(v12 + 24);
      if ( *(_BYTE *)v12 == 17 )
        goto LABEL_11;
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17 > 1 )
        goto LABEL_4;
      if ( *(_BYTE *)v12 > 0x15u )
        goto LABEL_4;
      v28 = (_BYTE *)sub_AD7630(v12, 0);
      if ( !v28 || *v28 != 17 )
        goto LABEL_4;
      v13 = v28 + 24;
LABEL_11:
      v14 = *((_DWORD *)v13 + 2);
      if ( v14 > 0x40 )
      {
        if ( v14 - (unsigned int)sub_C444A0(v13) <= 0x40 && !**(_QWORD **)v13 )
          goto LABEL_4;
        v30 = sub_C445E0(v13);
        LOBYTE(v10) = 1;
        if ( v14 != v30 )
          return (char)v10;
      }
      else
      {
        if ( !*(_QWORD *)v13 )
          goto LABEL_4;
        if ( v14 && *(_QWORD *)v13 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14) )
        {
LABEL_7:
          LOBYTE(v10) = 1;
          return (char)v10;
        }
      }
      v36 = 0;
      v35 = &v34;
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v15 = *(__int64 **)(a2 - 8);
      else
        v15 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v16 = *v15;
      if ( (unsigned __int8)sub_991580((__int64)&v35, *v15) )
      {
        LODWORD(v10) = sub_986B30(v34, v16, v17, v18, v19) ^ 1;
        return (char)v10;
      }
      goto LABEL_4;
    case 32:
      if ( !(_BYTE)a7 || *(_BYTE *)a2 != 61 || (unsigned __int8)sub_D30ED0(a2, (unsigned int)a7, (unsigned int)a8) )
        goto LABEL_4;
      v24 = sub_B43CC0(a2);
      _BitScanReverse64(&v25, 1LL << (*(_WORD *)(a2 + 2) >> 1));
      LOBYTE(v10) = sub_D305E0(
                      *(_QWORD *)(a2 - 32),
                      *(_QWORD *)(a2 + 8),
                      63 - ((unsigned int)v25 ^ 0x3F),
                      v24,
                      a3,
                      a4,
                      a5,
                      a6);
      return (char)v10;
    case 56:
      if ( *(_BYTE *)a2 != 85 )
        goto LABEL_4;
      if ( !(_BYTE)a8 )
      {
        LOBYTE(v10) = sub_CEA740(a2, (unsigned int)a7);
        return (char)v10;
      }
      v26 = *(_QWORD *)(a2 - 32);
      if ( !v26 || *(_BYTE *)v26 || *(_QWORD *)(v26 + 24) != *(_QWORD *)(a2 + 80) || (*(_BYTE *)(v26 + 33) & 0x20) == 0 )
        goto LABEL_4;
      v27 = *(_DWORD *)(v26 + 36);
      if ( v27 > 0x174 )
      {
        if ( v27 <= 0x23D9 )
        {
          LOBYTE(v10) = 0;
          if ( v27 >= 0x23D8 )
            LOBYTE(v10) = a8;
        }
        else
        {
          switch ( v27 )
          {
            case 0x245Bu:
            case 0x245Cu:
            case 0x245Du:
            case 0x248Bu:
            case 0x248Cu:
            case 0x248Du:
            case 0x2490u:
            case 0x2491u:
            case 0x2492u:
            case 0x249Au:
            case 0x249Bu:
            case 0x249Cu:
            case 0x249Eu:
LABEL_59:
              LOBYTE(v10) = a8;
              break;
            default:
              goto LABEL_4;
          }
        }
      }
      else
      {
        if ( v27 > 0x14C )
        {
          switch ( v27 )
          {
            case 0x14Du:
            case 0x153u:
            case 0x168u:
            case 0x171u:
            case 0x174u:
              goto LABEL_59;
            default:
              goto LABEL_4;
          }
        }
        if ( v27 > 0x47 )
        {
          LOBYTE(v10) = a8;
          if ( v27 != 282 )
          {
            LOBYTE(v10) = 0;
            if ( v27 == 312 )
              LOBYTE(v10) = a8;
          }
        }
        else
        {
          if ( v27 <= 0xE )
          {
LABEL_4:
            LOBYTE(v10) = 0;
            return (char)v10;
          }
          LOBYTE(v10) = 0;
          if ( ((1LL << ((unsigned __int8)v27 - 15)) & 0x15C000000000001LL) != 0 )
            LOBYTE(v10) = a8;
        }
      }
      return (char)v10;
    default:
      goto LABEL_7;
  }
}
