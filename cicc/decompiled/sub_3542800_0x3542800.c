// Function: sub_3542800
// Address: 0x3542800
//
__int64 __fastcall sub_3542800(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  unsigned __int8 v6; // dl
  __int16 v7; // si
  __int64 v8; // rsi
  __int64 v9; // rcx
  unsigned __int8 **v10; // r14
  unsigned __int8 **i; // r15
  unsigned __int8 *v12; // rdx
  _BYTE *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int8 *v17; // rsi
  unsigned __int8 *v18; // r13
  unsigned __int8 v19; // al
  unsigned __int8 *v20; // r9
  __int64 v21; // rdx
  unsigned __int8 *v22; // [rsp+8h] [rbp-38h]

  *(_BYTE *)(a1 + 568) = 0;
  *(_DWORD *)(a1 + 572) = 0;
  result = (__int64)sub_2EA6400(a2);
  if ( result )
  {
    result = *(_QWORD *)(result + 16);
    if ( result )
    {
      v4 = *(_QWORD *)(result + 48);
      result += 48;
      v5 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 != result )
      {
        if ( !v5 )
          BUG();
        result = (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30;
        if ( (unsigned int)result <= 0xA && (*(_BYTE *)(v5 - 17) & 0x20) != 0 )
        {
          result = sub_B91C10(v5 - 24, 18);
          if ( result )
          {
            v6 = *(_BYTE *)(result - 16);
            if ( (v6 & 2) != 0 )
            {
              v9 = *(_QWORD *)(result - 32);
              v8 = *(unsigned int *)(result - 24);
            }
            else
            {
              v7 = *(_WORD *)(result - 16) >> 6;
              result -= 8LL * ((v6 >> 2) & 0xF);
              v8 = v7 & 0xF;
              v9 = result - 16;
            }
            v10 = (unsigned __int8 **)(v9 + 8 * v8);
            for ( i = (unsigned __int8 **)(v9 + 8); v10 != i; ++i )
            {
              v17 = *i;
              result = (unsigned int)**i - 5;
              if ( (unsigned __int8)(**i - 5) <= 0x1Fu )
              {
                result = *(v17 - 16);
                v18 = v17 - 16;
                if ( (result & 2) != 0 )
                {
                  v12 = (unsigned __int8 *)*((_QWORD *)v17 - 4);
                }
                else
                {
                  result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
                  v12 = &v18[-result];
                }
                v13 = *(_BYTE **)v12;
                v22 = *i;
                if ( !**(_BYTE **)v12 )
                {
                  v14 = sub_B91420((__int64)v13);
                  if ( v15 == 37
                    && !(*(_QWORD *)v14 ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(v14 + 8) ^ 0x696C657069702E70LL)
                    && !(*(_QWORD *)(v14 + 16) ^ 0x6974696E692E656ELL | *(_QWORD *)(v14 + 24) ^ 0x746E696E6F697461LL)
                    && *(_DWORD *)(v14 + 32) == 1635152485
                    && *(_BYTE *)(v14 + 36) == 108 )
                  {
                    v19 = *(v22 - 16);
                    if ( (v19 & 2) != 0 )
                      v20 = (unsigned __int8 *)*((_QWORD *)v22 - 4);
                    else
                      v20 = &v18[-8 * ((v19 >> 2) & 0xF)];
                    v21 = *(_QWORD *)(*((_QWORD *)v20 + 1) + 136LL);
                    result = *(_QWORD *)(v21 + 24);
                    if ( *(_DWORD *)(v21 + 32) > 0x40u )
                      result = *(_QWORD *)result;
                    *(_DWORD *)(a1 + 572) = result;
                  }
                  else
                  {
                    result = sub_B91420((__int64)v13);
                    if ( v16 == 26
                      && !(*(_QWORD *)result ^ 0x6F6F6C2E6D766C6CLL | *(_QWORD *)(result + 8) ^ 0x696C657069702E70LL)
                      && *(_QWORD *)(result + 16) == 0x62617369642E656ELL
                      && *(_WORD *)(result + 24) == 25964 )
                    {
                      *(_BYTE *)(a1 + 568) = 1;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
