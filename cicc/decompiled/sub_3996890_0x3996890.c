// Function: sub_3996890
// Address: 0x3996890
//
__int64 __fastcall sub_3996890(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  int v7; // r15d
  __int64 v8; // rsi
  unsigned int v9; // r14d
  _BYTE *v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // r15
  unsigned int v15; // r14d
  unsigned int v16; // eax
  _BYTE *v17; // rcx
  unsigned int v18; // edx
  unsigned int v19; // esi
  unsigned int v20; // [rsp+Ch] [rbp-44h]
  __int64 v21; // [rsp+18h] [rbp-38h]

  sub_397F600((__int64 *)a1, a2);
  v3 = (__int64 *)sub_1E15F70(a2);
  result = sub_1626D20(*v3);
  if ( !result )
    return result;
  result = *(unsigned int *)(*(_QWORD *)(result + 8 * (5LL - *(unsigned int *)(result + 8))) + 36LL);
  if ( !(_DWORD)result )
    return result;
  result = **(unsigned __int16 **)(a2 + 16);
  switch ( **(_WORD **)(a2 + 16) )
  {
    case 2:
    case 3:
    case 4:
    case 6:
    case 9:
    case 0xC:
    case 0xD:
    case 0x11:
    case 0x12:
      return result;
    default:
      result = *(_QWORD *)(a2 + 16);
      if ( (*(_BYTE *)(a2 + 46) & 1) != 0 )
        return result;
      v5 = *(_QWORD *)(a1 + 24);
      v6 = a2 + 64;
      v7 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL) + 8LL) + 1028LL);
      result = *(_QWORD *)(a2 + 64);
      if ( v5 == result )
      {
        if ( !v7 )
        {
          if ( v5 )
          {
            result = sub_15C70B0(a2 + 64);
            if ( (_DWORD)result )
            {
              v14 = sub_15C70D0(a2 + 64);
              v15 = sub_15C70C0(a2 + 64);
              v16 = sub_15C70B0(a2 + 64);
              v17 = (_BYTE *)v14;
              v18 = v15;
              v19 = v16;
              return sub_398C0C0(a1, v19, v18, v17, 0);
            }
          }
        }
        return result;
      }
      if ( result )
      {
        if ( v5 )
        {
          result = v7 | (unsigned int)sub_15C70B0(a2 + 64);
          if ( !(_DWORD)result )
            return result;
          v8 = *(_QWORD *)(a1 + 48);
          v9 = 0;
          if ( *(_QWORD *)(a2 + 64) != v8 )
            goto LABEL_12;
          v21 = 0;
          if ( !v8 )
          {
LABEL_27:
            v9 = 5;
LABEL_12:
            if ( *(_QWORD *)(a1 + 24) )
              v7 = sub_15C70B0(a1 + 24);
            goto LABEL_14;
          }
        }
        else
        {
          v8 = *(_QWORD *)(a1 + 48);
          v9 = 0;
          if ( result != v8 )
          {
LABEL_14:
            if ( (unsigned int)sub_15C70B0(a2 + 64) && (unsigned int)sub_15C70B0(a2 + 64) != v7 )
              v9 |= 1u;
            v10 = (_BYTE *)sub_15C70D0(a2 + 64);
            if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL) + 504LL) - 34) <= 1
              && *(_DWORD *)(a1 + 6584) == 1 )
            {
              sub_3995FB0(a1, a2, v9);
            }
            else
            {
              v20 = sub_15C70C0(a2 + 64);
              v11 = sub_15C70B0(a2 + 64);
              sub_398C0C0(a1, v11, v20, v10, v9);
            }
            result = sub_15C70B0(a2 + 64);
            if ( (_DWORD)result && v6 != a1 + 24 )
            {
              v12 = *(_QWORD *)(a1 + 24);
              if ( v12 )
                result = sub_161E7C0(a1 + 24, v12);
              v13 = *(_QWORD *)(a2 + 64);
              *(_QWORD *)(a1 + 24) = v13;
              if ( v13 )
                return sub_1623A60(a1 + 24, v13, 2);
            }
            return result;
          }
          v21 = 0;
        }
        sub_161E7C0(a1 + 48, v8);
        *(_QWORD *)(a1 + 48) = v21;
        goto LABEL_27;
      }
      if ( v7 )
      {
        result = (unsigned int)dword_5056E40;
        if ( dword_5056E40 != 2
          && (dword_5056E40 == 1
           || *(_QWORD *)(a1 + 32)
           || (result = *(_QWORD *)(a1 + 40)) != 0 && result != *(_QWORD *)(a2 + 24)) )
        {
          v18 = 0;
          if ( v5 )
          {
            v5 = sub_15C70D0(a1 + 24);
            v18 = sub_15C70C0(a1 + 24);
          }
          v17 = (_BYTE *)v5;
          v19 = 0;
          return sub_398C0C0(a1, v19, v18, v17, 0);
        }
      }
      return result;
  }
}
