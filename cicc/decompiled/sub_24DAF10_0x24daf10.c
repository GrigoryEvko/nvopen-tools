// Function: sub_24DAF10
// Address: 0x24daf10
//
void __fastcall sub_24DAF10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v4; // eax
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 *v7; // rbx
  __int64 v8; // rsi
  __int64 v9; // rdi
  unsigned int v10; // [rsp-4Ch] [rbp-4Ch] BYREF
  __m128i v11; // [rsp-48h] [rbp-48h] BYREF
  __int64 v12; // [rsp-38h] [rbp-38h]

  v2 = *(_QWORD *)(a2 - 32);
  if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a2 + 80) )
  {
    v4 = *(_DWORD *)(v2 + 36);
    if ( v4 )
    {
      switch ( v4 )
      {
        case 238:
          if ( **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
          {
            v5 = *(_QWORD *)(a1 + 16);
            v11.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v11.m128i_i64[1] = a2;
            v12 = a2;
            v6 = *(_QWORD *)(v5 + 8);
            if ( v6 == *(_QWORD *)(v5 + 16) )
              goto LABEL_44;
            if ( !v6 )
              goto LABEL_13;
            goto LABEL_12;
          }
          break;
        case 240:
          if ( **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
          {
            v5 = *(_QWORD *)(a1 + 16);
            v11.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v11.m128i_i64[1] = a2;
            v12 = a2;
            v6 = *(_QWORD *)(v5 + 8);
            if ( v6 == *(_QWORD *)(v5 + 16) )
              goto LABEL_44;
            if ( !v6 )
              goto LABEL_13;
            goto LABEL_12;
          }
          break;
        case 241:
          if ( **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
          {
            v5 = *(_QWORD *)(a1 + 16);
            v11.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v11.m128i_i64[1] = a2;
            v12 = a2;
            v6 = *(_QWORD *)(v5 + 8);
            if ( v6 == *(_QWORD *)(v5 + 16) )
              goto LABEL_44;
            if ( !v6 )
              goto LABEL_13;
            goto LABEL_12;
          }
          break;
        case 243:
          if ( **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
          {
            v5 = *(_QWORD *)(a1 + 16);
            v11.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v11.m128i_i64[1] = a2;
            v12 = a2;
            v6 = *(_QWORD *)(v5 + 8);
            if ( v6 == *(_QWORD *)(v5 + 16) )
              goto LABEL_44;
            if ( !v6 )
              goto LABEL_13;
            goto LABEL_12;
          }
          break;
        case 245:
          if ( **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
          {
            v5 = *(_QWORD *)(a1 + 16);
            v11.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v11.m128i_i64[1] = a2;
            v12 = a2;
            v6 = *(_QWORD *)(v5 + 8);
            if ( v6 == *(_QWORD *)(v5 + 16) )
            {
LABEL_44:
              sub_24DABF0(v5, (_BYTE *)v6, &v11);
            }
            else
            {
              if ( v6 )
              {
LABEL_12:
                *(__m128i *)v6 = _mm_loadu_si128(&v11);
                *(_QWORD *)(v6 + 16) = v12;
                v6 = *(_QWORD *)(v5 + 8);
              }
LABEL_13:
              *(_QWORD *)(v5 + 8) = v6 + 24;
            }
          }
          break;
        default:
          sub_24DADD0(a1, a2);
          break;
      }
    }
    else if ( LOBYTE(qword_4FEC628[8]) )
    {
      if ( (v7 = *(__int64 **)(a1 + 8), !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23))
        && !(unsigned __int8)sub_B49560(a2, 23)
        || (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 4)
        || (unsigned __int8)sub_B49560(a2, 4) )
      {
        v8 = *(_QWORD *)(a2 - 32);
        if ( v8
          && !*(_BYTE *)v8
          && *(_QWORD *)(v8 + 24) == *(_QWORD *)(a2 + 80)
          && sub_981210(*v7, v8, &v10)
          && (v10 == 357 || v10 == 186)
          && **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
        {
          v9 = *(_QWORD *)(a1 + 16);
          v11.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
          v11.m128i_i64[1] = a2;
          v12 = a2;
          sub_24DAD90(v9, &v11);
        }
      }
    }
  }
}
