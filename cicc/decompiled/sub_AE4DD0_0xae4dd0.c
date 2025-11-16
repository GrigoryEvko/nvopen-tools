// Function: sub_AE4DD0
// Address: 0xae4dd0
//
__int64 __fastcall sub_AE4DD0(__int64 a1, __int64 a2, char a3)
{
  char v3; // bl
  __int64 v5; // rdx
  _DWORD *v6; // rdi
  _DWORD *v7; // rax
  unsigned int v8; // r9d
  _DWORD *v9; // r10
  __int64 result; // rax
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // rax
  char v13; // cl
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  _DWORD *v16; // rdi
  int v17; // r10d
  _DWORD *v18; // r9
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rax
  unsigned int v24; // edx
  __int64 v25; // rax
  int v26; // [rsp+Ch] [rbp-34h] BYREF
  __int64 v27; // [rsp+10h] [rbp-30h]
  __int64 v28; // [rsp+18h] [rbp-28h]

  while ( 2 )
  {
    v3 = a3;
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
        v27 = sub_9208B0(a1, a2);
        v28 = v5;
        v6 = *(_DWORD **)(a1 + 128);
        v26 = v27;
        v7 = sub_AE1180(v6, (__int64)&v6[2 * *(unsigned int *)(a1 + 136)], &v26);
        if ( v9 != v7 && v8 == *v7 )
          goto LABEL_30;
        result = 0xFFFFFFFFLL;
        if ( v8 <= 7 )
          return result;
        result = 0;
        v11 = (v8 >> 3) - 1LL;
        if ( !v11 )
          return result;
        _BitScanReverse64(&v12, v11);
        v13 = 64 - (v12 ^ 0x3F);
        goto LABEL_7;
      case 8:
        v22 = 0;
        if ( a3 )
          return sub_AE4360(a1, v22);
        return sub_AE4370(a1, v22);
      case 0xA:
        return 6;
      case 0xC:
        return sub_AE3FE0(a1, *(_DWORD *)(a2 + 8) >> 8);
      case 0xE:
        v22 = *(_DWORD *)(a2 + 8) >> 8;
        if ( a3 )
          return sub_AE4360(a1, v22);
        else
          return sub_AE4370(a1, v22);
      case 0xF:
        if ( (*(_BYTE *)(a2 + 9) & 2) != 0 )
        {
          result = 0;
          if ( a3 )
            return result;
          v23 = sub_AE4AC0(a1, a2);
          goto LABEL_23;
        }
        v23 = sub_AE4AC0(a1, a2);
        if ( !v3 )
        {
LABEL_23:
          v24 = *(unsigned __int8 *)(a1 + 481);
          goto LABEL_24;
        }
        v24 = *(unsigned __int8 *)(a1 + 480);
LABEL_24:
        result = *(unsigned __int8 *)(v23 + 16);
        if ( (unsigned __int8)result <= (unsigned __int8)v24 )
          return v24;
        return result;
      case 0x10:
        a2 = *(_QWORD *)(a2 + 24);
        continue;
      case 0x11:
      case 0x12:
        v27 = sub_9208B0(a1, a2);
        v28 = v15;
        v16 = *(_DWORD **)(a1 + 176);
        v26 = v27;
        v7 = sub_AE1180(v16, (__int64)&v16[2 * *(unsigned int *)(a1 + 184)], &v26);
        if ( v18 != v7 && v17 == *v7 )
        {
LABEL_30:
          if ( v3 )
            return *((unsigned __int8 *)v7 + 4);
          else
            return *((unsigned __int8 *)v7 + 5);
        }
        else
        {
          v19 = sub_9208B0(a1, a2) + 7;
          result = 0xFFFFFFFFLL;
          v20 = v19 >> 3;
          if ( v20 )
          {
            result = 0;
            v21 = v20 - 1;
            if ( v21 )
            {
              _BitScanReverse64(&v21, v21);
              v13 = 64 - (v21 ^ 0x3F);
LABEL_7:
              _BitScanReverse64(&v14, 1LL << v13);
              return 63 - ((unsigned int)v14 ^ 0x3F);
            }
          }
        }
        return result;
      case 0x14:
        v25 = sub_BCE9B0(a2);
        a3 = v3;
        a2 = v25;
        continue;
      default:
        BUG();
    }
  }
}
