// Function: sub_11F1F40
// Address: 0x11f1f40
//
_BYTE *__fastcall sub_11F1F40(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4)
{
  __int64 *v6; // r15
  __int64 v7; // r8
  __int64 v9; // r8
  __int64 v10; // rax
  const char *v11; // rax
  size_t v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rcx
  char v16; // r8
  char v17; // dl
  unsigned int v18; // edx
  int v19; // eax
  char v20; // al
  __int64 v21; // rcx
  char v22; // dl
  __int64 v23; // rax
  const char *v24; // rax
  size_t v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  char v28; // al
  __int64 v29; // [rsp+8h] [rbp-58h]
  __m128i v30; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v31; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v32; // [rsp+28h] [rbp-38h]

  v6 = (__int64 *)sub_B43CA0(a2);
  if ( (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 72) || (unsigned __int8)sub_B49560(a2, 72) )
    return 0;
  v7 = sub_11F1ED0(a1, a2, a3, (__int64)a4, v9);
  if ( !v7 )
  {
    switch ( a3 )
    {
      case 0x56u:
      case 0x57u:
        v21 = (__int64)a4;
        v22 = 0;
        return (_BYTE *)sub_11EEEA0(a1, a2, v22, v21);
      case 0x86u:
      case 0x87u:
        v21 = (__int64)a4;
        v22 = 1;
        return (_BYTE *)sub_11EEEA0(a1, a2, v22, v21);
      case 0xA0u:
      case 0xA2u:
      case 0xA7u:
      case 0xADu:
      case 0xC1u:
      case 0xCEu:
      case 0xE3u:
      case 0xE4u:
      case 0xECu:
      case 0x1B4u:
      case 0x1ECu:
        if ( !*(_BYTE *)(a1 + 80) )
          return 0;
        v10 = *(_QWORD *)(a2 - 32);
        if ( v10 )
        {
          if ( !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *(_QWORD *)(a2 + 80) )
            v7 = *(_QWORD *)(a2 - 32);
        }
        else
        {
          v7 = 0;
        }
        v11 = sub_BD5D20(v7);
        if ( !(unsigned __int8)sub_11E9B60(a1, v6, (__int64)v11, v12, v13, v14) )
          return 0;
        v15 = *(__int64 **)(a1 + 24);
        v16 = 1;
        v17 = 0;
        return (_BYTE *)sub_11DB650(a2, (__int64)a4, v17, v15, v16);
      case 0xA9u:
      case 0xAAu:
      case 0xABu:
      case 0xB2u:
      case 0xB3u:
      case 0xB4u:
      case 0xD0u:
      case 0xD1u:
      case 0xD2u:
      case 0x1B6u:
      case 0x1B7u:
      case 0x1B8u:
      case 0x1EAu:
      case 0x1EBu:
      case 0x1EFu:
        return (_BYTE *)sub_11E9E00(a1, a2, (__int64)a4);
      case 0xBDu:
      case 0xBEu:
      case 0xBFu:
        return (_BYTE *)sub_11E40C0(a1, (unsigned __int8 *)a2, (__int64)a4);
      case 0xC4u:
        v18 = 21;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0xCBu:
        v23 = *(_QWORD *)(a2 - 32);
        if ( v23 )
        {
          if ( !*(_BYTE *)v23 && *(_QWORD *)(v23 + 24) == *(_QWORD *)(a2 + 80) )
            v7 = *(_QWORD *)(a2 - 32);
        }
        else
        {
          v7 = 0;
        }
        v24 = sub_BD5D20(v7);
        if ( !(unsigned __int8)sub_11E9B60(a1, v6, (__int64)v24, v25, v26, v27) )
          return 0;
        v15 = *(__int64 **)(a1 + 24);
        v16 = 0;
        v17 = 1;
        return (_BYTE *)sub_11DB650(a2, (__int64)a4, v17, v15, v16);
      case 0xE7u:
      case 0xE8u:
      case 0xE9u:
        return (_BYTE *)sub_11EA2A0(a1, a2, (__int64)a4);
      case 0xEFu:
      case 0xF0u:
      case 0xF1u:
        v18 = 170;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0x102u:
        v18 = 172;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0x108u:
      case 0x109u:
      case 0x10Au:
      case 0x10Bu:
      case 0x10Cu:
      case 0x10Du:
        return (_BYTE *)sub_11E9C60(a1, a2, (__int64)a4);
      case 0x114u:
      case 0x115u:
      case 0x116u:
        return sub_11E4D30((_QWORD *)a1, a2, (__int64)a4);
      case 0x14Du:
      case 0x14Eu:
      case 0x14Fu:
      case 0x150u:
      case 0x151u:
      case 0x152u:
      case 0x153u:
      case 0x154u:
      case 0x155u:
      case 0x156u:
      case 0x15Au:
      case 0x15Bu:
      case 0x15Cu:
      case 0x15Du:
      case 0x15Eu:
        return (_BYTE *)sub_11EBE80(a1, a2, (__int64)a4);
      case 0x173u:
      case 0x174u:
      case 0x175u:
        v19 = *(_DWORD *)(a2 + 4);
        v30 = 0u;
        v20 = sub_98B0F0(*(_QWORD *)(a2 - 32LL * (v19 & 0x7FFFFFF)), &v30, 1u);
        v7 = 0;
        if ( !v20 )
          return (_BYTE *)v7;
        v32 = 1;
        v31 = 0;
        if ( !v30.m128i_i64[1] )
        {
          v32 = 32;
LABEL_30:
          v7 = (__int64)sub_AD9140(*(_QWORD *)(a2 + 8), 0, (__int64)&v31);
          goto LABEL_31;
        }
        v28 = sub_C94210(&v30, 0, &v31);
        v7 = 0;
        if ( !v28 )
          goto LABEL_30;
LABEL_31:
        if ( v32 > 0x40 && v31 )
        {
          v29 = v7;
          j_j___libc_free_0_0(v31);
          v7 = v29;
        }
        break;
      case 0x176u:
        v18 = 250;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0x182u:
      case 0x183u:
      case 0x184u:
        return (_BYTE *)sub_11EF7B0(a1, a2, (__int64)a4);
      case 0x197u:
      case 0x198u:
      case 0x199u:
        return (_BYTE *)sub_11E50A0(a1, a2, a4);
      case 0x19Au:
      case 0x19Bu:
      case 0x19Cu:
        return (_BYTE *)sub_11E5590(a1, a2);
      case 0x1A0u:
        v18 = 308;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0x1A4u:
        v18 = 309;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0x1A5u:
        v18 = 310;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      case 0x1C0u:
      case 0x1C1u:
      case 0x1C2u:
        return (_BYTE *)sub_11EE900(a1, a2, (unsigned int **)a4);
      case 0x1F4u:
        v18 = 355;
        return (_BYTE *)sub_11DAE50(a2, (__int64)a4, v18);
      default:
        return (_BYTE *)v7;
    }
  }
  return (_BYTE *)v7;
}
