// Function: sub_257C1F0
// Address: 0x257c1f0
//
_BOOL8 __fastcall sub_257C1F0(__int64 *a1, unsigned __int64 a2, _BYTE *a3)
{
  _BOOL4 v3; // r13d
  __int64 v4; // r12
  unsigned __int8 v5; // al
  unsigned __int8 v6; // cl
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __m128i v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-58h]
  char v23; // [rsp+1Fh] [rbp-41h] BYREF
  __m128i v24; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_BYTE *)v4;
  v6 = *(_BYTE *)v4 - 34;
  if ( v6 <= 0x33u )
  {
    v3 = ((0x8000000000041uLL >> v6) & 1) == 0;
    if ( ((0x8000000000041uLL >> v6) & 1) == 0 )
    {
      if ( (v5 & 0xFD) != 0x54 && v5 != 63 )
      {
        LOBYTE(v3) = (unsigned __int8)(v5 - 61) <= 1u;
        return v3;
      }
      goto LABEL_3;
    }
    if ( *(char *)(v4 + 7) < 0 )
    {
      v8 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
      if ( *(char *)(v4 + 7) < 0 )
      {
        v10 = (v8 + v9 - sub_BD2BC0(v4)) >> 4;
        v11 = 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
        v12 = v4 - v11;
        if ( !(_DWORD)v10 || *(char *)(v4 + 7) >= 0 )
        {
LABEL_18:
          if ( a2 >= v12 && a2 < (unsigned __int64)sub_24E54B0((unsigned __int8 *)v4) )
          {
            v19.m128i_i64[0] = sub_254C9B0(v4, (__int64)(a2 - (v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) >> 5);
            v20 = a1[1];
            v21 = *a1;
            v24 = v19;
            return (_BOOL4)sub_257BF90(v21, v20, &v24, 0, &v23, 0, 0);
          }
          return 1;
        }
        v22 = v4 - v11;
        v13 = sub_BD2BC0(v4);
        v15 = v13 + v14;
        if ( *(char *)(v4 + 7) >= 0 )
        {
          if ( (unsigned int)(v15 >> 4) )
            goto LABEL_24;
        }
        else if ( (unsigned int)((v15 - sub_BD2BC0(v4)) >> 4) )
        {
          if ( *(char *)(v4 + 7) < 0 )
          {
            v16 = (__int64)(a2 - v22) >> 5;
            if ( *(_DWORD *)(sub_BD2BC0(v4) + 8) <= (unsigned int)v16 )
            {
              if ( *(char *)(v4 + 7) >= 0 )
                BUG();
              v17 = sub_BD2BC0(v4);
              if ( *(_DWORD *)(v17 + v18 - 4) > (unsigned int)v16 )
                return v3;
            }
            goto LABEL_17;
          }
LABEL_24:
          BUG();
        }
      }
    }
LABEL_17:
    v12 = v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
    goto LABEL_18;
  }
  if ( (v5 & 0xFD) == 0x54 )
  {
LABEL_3:
    *a3 = 1;
    return 1;
  }
  if ( v5 == 30 )
    LOBYTE(v3) = (unsigned int)((char)sub_2509800((_QWORD *)(a1[1] + 72)) - 6) <= 1;
  else
    return 0;
  return v3;
}
