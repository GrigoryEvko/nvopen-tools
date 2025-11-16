// Function: sub_14AD280
// Address: 0x14ad280
//
__int64 __fastcall sub_14AD280(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  __int64 v3; // r12
  bool v5; // r14
  int v6; // ebx
  unsigned __int8 v7; // al
  __int64 v8; // rax
  __int16 v9; // ax
  __int64 *v11; // r12
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int8 v14; // dl
  __int64 v15; // rdi
  __int64 v16; // rax
  __m128i v17; // [rsp+0h] [rbp-60h] BYREF
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]

  v3 = a1;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
    return v3;
  v5 = a3 != 0;
  v6 = 0;
  while ( 1 )
  {
    v7 = *(_BYTE *)(v3 + 16);
    if ( v7 <= 0x17u )
      break;
    if ( v7 == 56 || (unsigned __int8)(v7 - 71) <= 1u )
      goto LABEL_15;
    switch ( v7 )
    {
      case 0x35u:
        return v3;
      case 0x4Eu:
        v12 = v3 | 4;
        break;
      case 0x1Du:
        v12 = v3 & 0xFFFFFFFFFFFFFFFBLL;
        break;
      default:
        goto LABEL_9;
    }
    v13 = v12 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v13 )
      goto LABEL_9;
    v14 = *(_BYTE *)(v13 + 16);
    v15 = 0;
    if ( v14 > 0x17u )
    {
      if ( v14 == 78 )
      {
        v15 = v13 | 4;
      }
      else if ( v14 == 29 )
      {
        v15 = v13;
      }
    }
    v16 = sub_14AD130(v15);
    if ( !v16 )
    {
      if ( *(_BYTE *)(v3 + 16) <= 0x17u )
        return v3;
LABEL_9:
      v17 = (__m128i)a2;
      v18 = 0;
      v19 = 0;
      v20 = v3;
      v8 = sub_13E3350(v3, &v17, 0, 1, (__int64)&unk_428F1B4);
      if ( !v8 )
        return v3;
      v3 = v8;
      goto LABEL_18;
    }
    v3 = v16;
LABEL_18:
    if ( a3 <= ++v6 && v5 )
      return v3;
  }
  if ( v7 == 5 )
  {
    v9 = *(_WORD *)(v3 + 18);
    if ( v9 != 32 && (unsigned __int16)(v9 - 47) > 1u )
      return v3;
LABEL_15:
    if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
      v11 = *(__int64 **)(v3 - 8);
    else
      v11 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
    v3 = *v11;
    goto LABEL_18;
  }
  if ( v7 == 1 )
    __asm { jmp     rax }
  return v3;
}
