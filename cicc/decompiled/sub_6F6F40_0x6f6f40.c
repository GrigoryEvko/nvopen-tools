// Function: sub_6F6F40
// Address: 0x6f6f40
//
__int64 __fastcall sub_6F6F40(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r13
  char v9; // al
  __int8 v10; // al
  __int64 v12; // rbx
  __int64 v13; // r14
  _QWORD *v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r14
  char v22; // al
  _DWORD v23[9]; // [rsp+Ch] [rbp-24h] BYREF

  v7 = sub_6F6EB0((__int64)a1, a2, a3, a4, a5, a6);
  v8 = v7;
  if ( (a1[1].m128i_i8[4] & 4) == 0 )
    goto LABEL_7;
  v9 = *(_BYTE *)(v7 + 24);
  if ( v9 == 20 || v9 == 3 )
    goto LABEL_13;
  if ( v9 != 2 )
  {
    if ( v9 != 4 )
    {
      if ( v9 == 1 && (unsigned __int8)(*(_BYTE *)(v8 + 56) - 22) <= 1u )
      {
        v13 = a1[2].m128i_i64[1];
        if ( v13 )
        {
          if ( *(_QWORD *)(v13 + 8) )
          {
            sub_7296F0((unsigned int)dword_4F04C64, v23);
            v14 = (_QWORD *)sub_726700(20);
            *v14 = sub_72CBE0(20, v23, v15, v16, v17, v18);
            v19 = sub_7CADA0(v13, &a1[1].m128i_u64[1]);
            v20 = v23[0];
            v14[8] = v19;
            *(_QWORD *)(*(_QWORD *)(v8 + 72) + 16LL) = v14;
            sub_729730(v20);
          }
        }
      }
      goto LABEL_7;
    }
LABEL_13:
    *(_QWORD *)(v8 + 64) = sub_7CADA0(*(_QWORD *)(v8 + 56), &a1[1].m128i_u64[1]);
    goto LABEL_7;
  }
  v21 = *(_QWORD *)(v8 + 56);
  if ( (unsigned int)sub_72AE00(v21)
    || *(_BYTE *)(v21 + 173) == 12
    && ((v22 = *(_BYTE *)(v21 + 176), (unsigned __int8)(v22 - 2) <= 2u) || ((v22 - 11) & 0xFD) == 0) )
  {
    *(_QWORD *)(v8 + 64) = sub_7CADA0(v21, &a1[1].m128i_u64[1]);
  }
LABEL_7:
  if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    v10 = a1[1].m128i_i8[0];
    if ( v10 == 1 )
    {
      v12 = a1[9].m128i_i64[0];
    }
    else
    {
      if ( v10 != 2 )
      {
LABEL_11:
        sub_6E39C0(a1, v8);
        return v8;
      }
      v12 = a1[18].m128i_i64[0];
      if ( !v12 )
      {
        if ( a1[19].m128i_i8[13] != 12 || a1[20].m128i_i8[0] != 1 )
          goto LABEL_11;
        v12 = sub_72E9A0(&a1[9]);
      }
    }
    sub_6E39C0(a1, v8);
    if ( v12 && v12 != v8 && !*(_QWORD *)(v12 + 80) )
      *(_QWORD *)(v12 + 80) = *(_QWORD *)(v8 + 80);
  }
  return v8;
}
