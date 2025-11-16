// Function: sub_1D14420
// Address: 0x1d14420
//
__int64 __fastcall sub_1D14420(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 v3; // r12
  int v4; // eax
  __int64 v6; // rsi
  char v7; // al
  __int64 v8; // rdx
  int v9; // r15d
  __int64 v10; // r14
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 v13; // rax
  __int64 v14; // r13
  int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // rsi
  bool v18; // al
  char v19; // dl
  __int64 v20; // rsi
  __int64 v21; // rax
  char v23; // al
  char v24; // al
  __int64 v25; // rdx
  unsigned int v26; // [rsp+4h] [rbp-4Ch]
  unsigned int v27; // [rsp+8h] [rbp-48h]
  char v28[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v29; // [rsp+18h] [rbp-38h]

  v3 = a1;
  v4 = *(unsigned __int16 *)(a1 + 24);
  if ( v4 != 10 && v4 != 32 )
  {
    if ( v4 != 104 )
      return 0;
    v6 = *(_QWORD *)(a1 + 40) + 16LL * a2;
    v7 = *(_BYTE *)v6;
    v8 = *(_QWORD *)(v6 + 8);
    v28[0] = v7;
    v29 = v8;
    v9 = v7 ? word_42E7700[(unsigned __int8)(v7 - 14)] : sub_1F58D30(v28);
    if ( !v9 )
      return 0;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    while ( 1 )
    {
      v13 = *a3;
      if ( *((_DWORD *)a3 + 2) > 0x40u )
        v13 = *(_QWORD *)(v13 + 8LL * (v12 >> 6));
      if ( (v13 & (1LL << v12)) != 0 )
        break;
LABEL_23:
      ++v12;
      v10 += 40;
      if ( v12 == v9 )
        return v11;
    }
    v14 = *(_QWORD *)(*(_QWORD *)(v3 + 32) + v10);
    v15 = *(unsigned __int16 *)(v14 + 24);
    if ( v15 != 32 && v15 != 10 )
      return 0;
    if ( v11 )
    {
      v16 = *(_QWORD *)(v11 + 88);
      v17 = *(_QWORD *)(v14 + 88);
      if ( *(_DWORD *)(v16 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(v16 + 24) != *(_QWORD *)(v17 + 24) )
          return 0;
      }
      else
      {
        v27 = v12;
        v18 = sub_16A5220(v16 + 24, (const void **)(v17 + 24));
        v12 = v27;
        if ( !v18 )
          return 0;
      }
    }
    v19 = v28[0];
    if ( v28[0] )
    {
      if ( (unsigned __int8)(v28[0] - 14) <= 0x5Fu )
      {
        switch ( v28[0] )
        {
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
            v19 = 3;
            v20 = 0;
            break;
          case 0x21:
          case 0x22:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x28:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
            v19 = 4;
            v20 = 0;
            break;
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
            v19 = 5;
            v20 = 0;
            break;
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x55:
            v19 = 6;
            v20 = 0;
            break;
          case 0x37:
            v19 = 7;
            v20 = 0;
            break;
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x62:
          case 0x63:
          case 0x64:
            v19 = 8;
            v20 = 0;
            break;
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x65:
          case 0x66:
          case 0x67:
          case 0x68:
          case 0x69:
            v19 = 9;
            v20 = 0;
            break;
          case 0x5E:
          case 0x5F:
          case 0x60:
          case 0x61:
          case 0x6A:
          case 0x6B:
          case 0x6C:
          case 0x6D:
            v19 = 10;
            v20 = 0;
            break;
          default:
            v19 = 2;
            v20 = 0;
            break;
        }
        goto LABEL_19;
      }
    }
    else
    {
      v26 = v12;
      v23 = sub_1F58D20(v28);
      v12 = v26;
      v19 = 0;
      if ( v23 )
      {
        v24 = sub_1F596B0(v28);
        v12 = v26;
        v20 = v25;
        v19 = v24;
LABEL_19:
        v21 = *(_QWORD *)(v14 + 40);
        if ( *(_BYTE *)v21 != v19 || *(_QWORD *)(v21 + 8) != v20 && !v19 )
          return 0;
        v11 = v14;
        goto LABEL_23;
      }
    }
    v20 = v29;
    goto LABEL_19;
  }
  return v3;
}
