// Function: sub_1AE93B0
// Address: 0x1ae93b0
//
__int64 __fastcall sub_1AE93B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v13; // rsi
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // [rsp+10h] [rbp-60h]
  __int64 v18; // [rsp+20h] [rbp-50h]
  unsigned __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int8 v20; // [rsp+38h] [rbp-38h]

  v2 = a1;
  v3 = 1;
  v4 = sub_15F2050(a2);
  v5 = sub_1632FA0(v4);
  v6 = (unsigned int)sub_15A9FE0(v5, a1);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v15 = *(_QWORD *)(v2 + 32);
        v2 = *(_QWORD *)(v2 + 24);
        v3 *= v15;
        continue;
      case 1:
        v7 = 16;
        goto LABEL_5;
      case 2:
        v7 = 32;
        goto LABEL_5;
      case 3:
      case 9:
        v7 = 64;
        goto LABEL_5;
      case 4:
        v7 = 80;
        goto LABEL_5;
      case 5:
      case 6:
        v7 = 128;
        goto LABEL_5;
      case 7:
        v7 = 8 * (unsigned int)sub_15A9520(v5, 0);
        goto LABEL_5;
      case 0xB:
        v7 = *(_DWORD *)(v2 + 8) >> 8;
        goto LABEL_5;
      case 0xD:
        v7 = 8LL * *(_QWORD *)sub_15A9930(v5, v2);
        goto LABEL_5;
      case 0xE:
        v18 = *(_QWORD *)(v2 + 24);
        sub_15A9FE0(v5, v18);
        v13 = v18;
        v14 = 1;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v13 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v16 = *(_QWORD *)(v13 + 32);
              v13 = *(_QWORD *)(v13 + 24);
              v14 *= v16;
              continue;
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 9:
            case 0xB:
              goto LABEL_24;
            case 7:
              sub_15A9520(v5, 0);
              goto LABEL_24;
            case 0xD:
              sub_15A9930(v5, v13);
              goto LABEL_24;
            case 0xE:
              v17 = *(_QWORD *)(v13 + 24);
              sub_15A9FE0(v5, v17);
              sub_127FA20(v5, v17);
              goto LABEL_24;
            case 0xF:
              sub_15A9520(v5, *(_DWORD *)(v13 + 8) >> 8);
LABEL_24:
              JUMPOUT(0x1AE95DE);
          }
        }
      case 0xF:
        v7 = 8 * (unsigned int)sub_15A9520(v5, *(_DWORD *)(v2 + 8) >> 8);
LABEL_5:
        v8 = 8 * v6 * ((v6 + ((unsigned __int64)(v3 * v7 + 7) >> 3) - 1) / v6);
        sub_1601A80((__int64)&v19, a2);
        v9 = v20;
        if ( v20 )
          goto LABEL_23;
        v10 = *(_QWORD *)(a2 - 24);
        if ( *(_BYTE *)(v10 + 16) )
          BUG();
        if ( *(_DWORD *)(v10 + 36) != 38 )
        {
          v11 = sub_1601A30(a2, 1);
          if ( v11 )
          {
            if ( *(_BYTE *)(v11 + 16) == 53 )
            {
              sub_15F8C50((__int64)&v19, v11, v5);
              v9 = v20;
              if ( v20 )
LABEL_23:
                LOBYTE(v9) = v19 <= v8;
            }
          }
        }
        return v9;
    }
  }
}
