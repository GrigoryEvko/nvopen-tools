// Function: sub_84A490
// Address: 0x84a490
//
__int64 __fastcall sub_84A490(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // r13
  bool v7; // r13
  _DWORD *v8; // rax
  __int64 v9; // rdi
  __m128i *v10; // r8
  __m128i *v11; // r9
  _BOOL4 v12; // ecx
  _BYTE *v13; // rdx
  __int64 v14; // r8
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _DWORD *v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r9
  __m128i *v25; // [rsp+0h] [rbp-1A0h]
  _BOOL4 v26; // [rsp+8h] [rbp-198h]
  __m128i v27[25]; // [rsp+10h] [rbp-190h] BYREF

  if ( a1 )
  {
    v6 = *(_BYTE *)(qword_4D03C50 + 21LL);
    *(_BYTE *)(qword_4D03C50 + 21LL) = v6 | 4;
    v7 = (v6 & 4) != 0;
    v8 = (_DWORD *)sub_6E1A20(a1);
    sub_82F0D0(a2, v8);
    if ( *(_DWORD *)(a2 + 8) != 6 )
    {
      if ( *(_BYTE *)(a1 + 8) != 1 )
      {
        v9 = *(_QWORD *)(a1 + 24);
        v10 = (__m128i *)(v9 + 8);
        if ( (*(_BYTE *)(a1 + 9) & 8) != 0 )
        {
          if ( a3 )
          {
            v11 = v27;
            v12 = dword_4F077BC == 0;
            while ( 1 )
            {
              v25 = v11;
              v26 = v12;
              sub_827F00(v9 + 8, *(_QWORD *)(a3 + 8), 0, v12, !v12, v11);
              v11 = v25;
              if ( !v26 )
                break;
              v12 = 0;
              if ( !v27[0].m128i_i32[0] )
                break;
              v9 = *(_QWORD *)(a1 + 24);
            }
            v13 = (_BYTE *)(a2 + 48);
            v10 = (__m128i *)(*(_QWORD *)(a1 + 24) + 8LL);
            goto LABEL_14;
          }
        }
        else
        {
          v13 = (_BYTE *)(a2 + 48);
          if ( a3 )
          {
LABEL_14:
            v16 = a3;
            sub_843D70(v10, a3, v13, 0xA7u);
LABEL_15:
            v14 = sub_6F7150((const __m128i *)(*(_QWORD *)(a1 + 24) + 8LL), v16, v17, v18, v19, v20);
LABEL_16:
            *(_BYTE *)(qword_4D03C50 + 21LL) = *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFB | (4 * v7);
            return v14;
          }
        }
        v16 = 1;
        sub_6FE880((__m128i *)(v9 + 8), 1);
        goto LABEL_15;
      }
      if ( a3 )
      {
        sub_848800(a1, a3, (_BYTE *)(a2 + 48), 0xA7u, v27);
        v14 = sub_6F7150(v27, a3, v22, v23, (__int64)v27, v24);
        goto LABEL_16;
      }
      v21 = (_DWORD *)sub_6E1A20(a1);
      if ( (unsigned int)sub_6E5430() )
        sub_6851C0(0x92Fu, v21);
    }
    v14 = (__int64)sub_7305B0();
    goto LABEL_16;
  }
  v14 = *(_QWORD *)(a3 + 40);
  if ( v14 || (*(_BYTE *)(a3 + 32) & 0x10) != 0 )
    return sub_73F4A0(
             a4,
             a3,
             (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) != 0,
             (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0,
             *(_BYTE *)(qword_4D03C50 + 17LL) & 1);
  return v14;
}
