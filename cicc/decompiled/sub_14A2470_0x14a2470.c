// Function: sub_14A2470
// Address: 0x14a2470
//
__int64 __fastcall sub_14A2470(__int64 a1, __int64 a2, int a3)
{
  int v3; // r14d
  unsigned int v4; // ebx
  __int64 v5; // rax
  _BYTE *v6; // r8
  _BYTE *v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  const void *v10; // r14
  signed __int64 v11; // r12
  const void *v12; // r15
  unsigned int v13; // r12d
  _BYTE *v15; // [rsp+0h] [rbp-80h] BYREF
  __int64 v16; // [rsp+8h] [rbp-78h]
  _BYTE dest[112]; // [rsp+10h] [rbp-70h] BYREF

  v3 = a3;
  if ( a3 < 0 )
    v3 = *(_DWORD *)(a2 + 96);
  v4 = *(_DWORD *)(a2 + 36);
  if ( v4 )
  {
    v5 = *(_QWORD *)(a2 + 24);
    v15 = dest;
    v6 = dest;
    v7 = dest;
    v8 = *(_QWORD *)(v5 + 16);
    v9 = *(unsigned int *)(v5 + 12);
    v16 = 0x800000000LL;
    v9 *= 8;
    v10 = (const void *)(v8 + 8);
    v11 = v9 - 8;
    v12 = (const void *)(v8 + v9);
    if ( (unsigned __int64)(v9 - 8) > 0x40 )
    {
      sub_16CD150(&v15, dest, v11 >> 3, 8);
      v7 = v15;
      v6 = &v15[8 * (unsigned int)v16];
    }
    if ( v10 != v12 )
    {
      memcpy(v6, v10, v11);
      v7 = v15;
    }
    if ( v4 <= 0x1189 )
    {
      if ( v4 > 0x1182 )
      {
        v13 = ((1LL << ((unsigned __int8)v4 + 125)) & 0x49) == 0 ? 1 : 4;
LABEL_14:
        if ( v7 != dest )
          _libc_free((unsigned __int64)v7);
        return v13;
      }
      if ( v4 > 0x95 )
      {
        if ( v4 == 191 )
LABEL_24:
          v13 = 0;
        else
          v13 = v4 != 215;
        goto LABEL_14;
      }
      if ( v4 > 2 )
      {
        switch ( v4 )
        {
          case 3u:
          case 4u:
          case 0xEu:
          case 0xFu:
          case 0x12u:
          case 0x13u:
          case 0x14u:
          case 0x17u:
          case 0x1Bu:
          case 0x1Cu:
          case 0x1Du:
          case 0x24u:
          case 0x25u:
          case 0x26u:
          case 0x4Cu:
          case 0x4Du:
          case 0x71u:
          case 0x72u:
          case 0x74u:
          case 0x75u:
          case 0x90u:
          case 0x95u:
            goto LABEL_24;
          default:
            break;
        }
      }
    }
    v13 = 1;
    goto LABEL_14;
  }
  v13 = 1;
  if ( sub_14A2090(a1, (_BYTE *)a2) )
  {
    if ( v3 < 0 )
      v3 = *(_DWORD *)(*(_QWORD *)(a2 + 24) + 12LL) - 1;
    return (unsigned int)(v3 + 1);
  }
  return v13;
}
