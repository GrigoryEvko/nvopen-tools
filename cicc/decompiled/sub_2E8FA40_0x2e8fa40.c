// Function: sub_2E8FA40
// Address: 0x2e8fa40
//
__int64 __fastcall sub_2E8FA40(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // eax
  __int64 result; // rax
  _BYTE *v5; // rbx
  _BYTE *v6; // r12
  _BYTE *v7; // r15
  _BYTE *v8; // rbx
  __m128i v9; // [rsp+0h] [rbp-60h] BYREF
  __int64 v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h]

  if ( a2 - 1 > 0x3FFFFFFE )
  {
    v5 = *(_BYTE **)(a1 + 32);
    v6 = &v5[40 * (*(_DWORD *)(a1 + 40) & 0xFFFFFF)];
    if ( v5 != v6 )
    {
      v7 = *(_BYTE **)(a1 + 32);
      result = sub_2DADC00(v7);
      if ( (_BYTE)result )
        goto LABEL_10;
      while ( 1 )
      {
        v5 += 40;
        if ( v6 == v5 )
          break;
        v7 = v5;
        result = sub_2DADC00(v5);
        if ( (_BYTE)result )
        {
LABEL_10:
          while ( v6 != v7 )
          {
            if ( a2 == *((_DWORD *)v7 + 2) && (*(_DWORD *)v7 & 0xFFF00) == 0 )
              return result;
            v8 = v7 + 40;
            if ( v7 + 40 == v6 )
              goto LABEL_4;
            while ( 1 )
            {
              v7 = v8;
              result = sub_2DADC00(v8);
              if ( (_BYTE)result )
                break;
              v8 += 40;
              if ( v6 == v8 )
                goto LABEL_4;
            }
          }
          goto LABEL_4;
        }
      }
    }
    goto LABEL_4;
  }
  v3 = sub_2E8E710(a1, a2, a3, 0, 0);
  if ( v3 == -1 || (result = *(_QWORD *)(a1 + 32) + 40LL * v3) == 0 )
  {
LABEL_4:
    v9.m128i_i32[2] = a2;
    v9.m128i_i64[0] = 805306368;
    v10 = 0;
    v11 = 0;
    v12 = 0;
    return sub_2E8F270(a1, &v9);
  }
  return result;
}
