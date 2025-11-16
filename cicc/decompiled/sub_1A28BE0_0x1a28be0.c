// Function: sub_1A28BE0
// Address: 0x1a28be0
//
char __fastcall sub_1A28BE0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // r9d
  unsigned __int64 v15; // rsi
  __int64 v17[6]; // [rsp+0h] [rbp-30h] BYREF

  v6 = a1;
  LOBYTE(v7) = *(_BYTE *)(a2 + 16) - 24;
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x35:
    case 0x39:
    case 0x3A:
    case 0x3B:
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x46:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x53:
    case 0x54:
    case 0x55:
    case 0x56:
    case 0x57:
    case 0x58:
      goto LABEL_2;
    case 0x1D:
      v15 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a1 + 16) = v15 | *(_QWORD *)(a1 + 16) & 3LL | 4;
      v7 = *(_QWORD *)(a1 + 8) & 3LL;
      *(_QWORD *)(a1 + 8) = v7 | v15 | 4;
      return v7;
    case 0x36:
      LOBYTE(v7) = sub_15F32D0(a2);
      if ( ((_BYTE)v7 || (*(_BYTE *)(a2 + 18) & 1) != 0 || (LOBYTE(v7) = sub_1A1E0B0(*(_QWORD *)a2), (_BYTE)v7))
        && *(_BYTE *)(a1 + 344) )
      {
        v11 = sub_15F2050(a2);
        v12 = sub_1632FA0(v11);
        v13 = sub_127FA20(v12, *(_QWORD *)a2);
        LOBYTE(v7) = sub_1A22CF0(
                       (_QWORD *)a1,
                       a2,
                       a1 + 352,
                       (unsigned __int64)(v13 + 7) >> 3,
                       !(*(_BYTE *)(a2 + 18) & 1 | (*(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11)),
                       v14);
      }
      else
      {
LABEL_2:
        *(_QWORD *)(a1 + 8) = a2 | *(_QWORD *)(a1 + 8) & 3LL | 4;
      }
      return v7;
    case 0x37:
      LOBYTE(v7) = (unsigned __int8)sub_1A23310(a1, a2);
      return v7;
    case 0x38:
      LOBYTE(v7) = sub_1A21C80(a1, a2, a3, a4, a5, a6);
      return v7;
    case 0x45:
      *(_QWORD *)(a1 + 16) = a2 | *(_QWORD *)(a1 + 16) & 3LL | 4;
      return v7;
    case 0x47:
      if ( !*(_QWORD *)(a2 + 8) )
        goto LABEL_11;
      goto LABEL_9;
    case 0x48:
      if ( *(_QWORD *)(a2 + 8) )
      {
        v8 = *(_QWORD *)a2;
        v9 = *(_QWORD *)a1;
        if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
          v8 = **(_QWORD **)(v8 + 16);
        v10 = sub_15A9520(v9, *(_DWORD *)(v8 + 8) >> 8);
        sub_16A5D10((__int64)v17, v6 + 352, 8 * v10);
        sub_1A1A780((__int64 *)(v6 + 352), v17);
        sub_135E100(v17);
        a1 = v6;
LABEL_9:
        LOBYTE(v7) = sub_386EA80(a1, a2);
      }
      else
      {
LABEL_11:
        LOBYTE(v7) = sub_1A21B40(a1, a2, a3, a4, a5, a6);
      }
      return v7;
    case 0x4D:
    case 0x4F:
      LOBYTE(v7) = sub_1A28880(a1, a2, a3, a4, a5, a6);
      return v7;
    case 0x4E:
      v7 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v7 + 16) )
        goto LABEL_27;
      LODWORD(v7) = *(_DWORD *)(v7 + 36);
      if ( (_DWORD)v7 == 135 )
        goto LABEL_30;
      if ( (unsigned int)v7 > 0x87 )
      {
        if ( (_DWORD)v7 == 137 )
        {
          LOBYTE(v7) = sub_1A22EA0(a1, a2, a3, a4, a5, a6);
          return v7;
        }
LABEL_28:
        LOBYTE(v7) = sub_1A23110(a1, a2, a3, a4, a5, a6);
        return v7;
      }
      if ( (unsigned int)v7 <= 0x26 )
      {
        if ( (unsigned int)v7 > 0x23 )
          return v7;
        if ( !(_DWORD)v7 )
        {
LABEL_27:
          v7 = a2 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(a1 + 8) & 3LL | 4;
          *(_QWORD *)(a1 + 16) = a2 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)(a1 + 16) & 3LL | 4;
          *(_QWORD *)(a1 + 8) = v7;
          return v7;
        }
        goto LABEL_28;
      }
      if ( (_DWORD)v7 != 133 )
        goto LABEL_28;
LABEL_30:
      LOBYTE(v7) = (unsigned __int8)sub_1A27E90(a1, a2, a3, a4, a5, a6);
      return v7;
  }
}
