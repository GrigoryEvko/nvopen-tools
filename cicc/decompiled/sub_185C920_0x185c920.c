// Function: sub_185C920
// Address: 0x185c920
//
__int64 __fastcall sub_185C920(__int64 a1, __int64 a2)
{
  __int64 i; // r14
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int8 v8; // al
  __int64 result; // rax
  __int64 v10; // rax
  __int64 *v11; // rax
  char v12; // dl
  __int64 v13; // r12
  __int64 *v14; // rsi
  unsigned int v15; // edi
  __int64 *v16; // rcx

  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v5 = sub_1648700(i);
    v6 = (__int64)v5;
    if ( *((_BYTE *)v5 + 16) <= 0x17u )
      return 0;
    v7 = sub_15F2060((__int64)v5);
    if ( sub_15E4690(v7, 0) )
      return 0;
    v8 = *(_BYTE *)(v6 + 16);
    if ( v8 <= 0x17u )
      return 0;
    if ( v8 != 54 )
    {
      switch ( v8 )
      {
        case 0x37u:
          v10 = *(_QWORD *)(v6 - 48);
          if ( v10 && a1 == v10 )
            return 0;
          break;
        case 0x4Eu:
          if ( a1 != *(_QWORD *)(v6 - 24) )
            return 0;
          break;
        case 0x1Du:
          if ( a1 != *(_QWORD *)(v6 - 72) )
            return 0;
          break;
        case 0x47u:
        case 0x38u:
          if ( !(unsigned __int8)sub_185C920(v6, a2) )
            return 0;
          break;
        case 0x4Du:
          v11 = *(__int64 **)(a2 + 8);
          if ( *(__int64 **)(a2 + 16) != v11 )
            goto LABEL_21;
          v14 = &v11[*(unsigned int *)(a2 + 28)];
          v15 = *(_DWORD *)(a2 + 28);
          if ( v11 != v14 )
          {
            v16 = 0;
            while ( v6 != *v11 )
            {
              if ( *v11 == -2 )
                v16 = v11;
              if ( v14 == ++v11 )
              {
                if ( !v16 )
                  goto LABEL_41;
                *v16 = v6;
                --*(_DWORD *)(a2 + 32);
                ++*(_QWORD *)a2;
                goto LABEL_22;
              }
            }
            continue;
          }
LABEL_41:
          if ( v15 < *(_DWORD *)(a2 + 24) )
          {
            *(_DWORD *)(a2 + 28) = v15 + 1;
            *v14 = v6;
            ++*(_QWORD *)a2;
          }
          else
          {
LABEL_21:
            sub_16CCBA0(a2, v6);
            if ( !v12 )
              continue;
          }
LABEL_22:
          result = sub_185C920(v6, a2);
          if ( !(_BYTE)result )
            return result;
          break;
        case 0x4Bu:
          v13 = (*(_BYTE *)(v6 + 23) & 0x40) != 0 ? *(_QWORD *)(v6 - 8) : v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
          if ( *(_BYTE *)(*(_QWORD *)(v13 + 24) + 16LL) != 15 )
            return 0;
          break;
        default:
          return 0;
      }
    }
  }
  return 1;
}
