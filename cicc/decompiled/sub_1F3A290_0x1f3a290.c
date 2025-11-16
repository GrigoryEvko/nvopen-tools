// Function: sub_1F3A290
// Address: 0x1f3a290
//
_QWORD *__fastcall sub_1F3A290(__int64 a1, __int64 a2, char *a3, __int64 a4, int a5, _QWORD *a6)
{
  __int16 v7; // ax
  unsigned int v8; // ebx
  unsigned int *v9; // rax
  _QWORD *v10; // r13
  __int64 v12; // r12
  __int64 v13; // rdx
  unsigned int v14; // r12d
  _DWORD *v15; // r8
  char *v16; // rax
  signed __int64 v17; // rdx
  char *v21; // [rsp+28h] [rbp-88h]
  __int64 v23; // [rsp+38h] [rbp-78h]
  unsigned int v24; // [rsp+48h] [rbp-68h] BYREF
  unsigned int v25; // [rsp+4Ch] [rbp-64h] BYREF
  __m128i v26; // [rsp+50h] [rbp-60h] BYREF
  __int64 v27; // [rsp+60h] [rbp-50h]
  __int64 v28; // [rsp+68h] [rbp-48h]

  v7 = **(_WORD **)(a2 + 16);
  if ( v7 == 21 )
  {
    sub_211A5B0(&v26);
    v8 = *(_DWORD *)(*(_QWORD *)(v26.m128i_i64[0] + 32) + 40LL * ((unsigned int)v26.m128i_u8[8] + 3) + 24)
       + v26.m128i_u8[8]
       + 5;
LABEL_4:
    v23 = 4 * a4;
    v21 = &a3[4 * a4];
    if ( a3 != v21 )
      goto LABEL_5;
LABEL_10:
    v10 = sub_1E0B640(a1, a6[1] + ((unsigned __int64)**(unsigned __int16 **)(a2 + 16) << 6), (__int64 *)(a2 + 64), 1u);
    if ( !v8 )
      goto LABEL_13;
    goto LABEL_11;
  }
  if ( v7 == 23 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a2 + 32) + 104LL) + 4;
    goto LABEL_4;
  }
  sub_211A5A0(&v26);
  v23 = 4 * a4;
  v21 = &a3[4 * a4];
  if ( a3 == v21 )
  {
    v8 = 2;
    v10 = sub_1E0B640(a1, a6[1] + ((unsigned __int64)**(unsigned __int16 **)(a2 + 16) << 6), (__int64 *)(a2 + 64), 1u);
LABEL_11:
    v12 = 0;
    do
    {
      v13 = 5 * v12++;
      sub_1E1A9C0((__int64)v10, a1, (const __m128i *)(*(_QWORD *)(a2 + 32) + 8 * v13));
    }
    while ( v8 > (unsigned int)v12 );
LABEL_13:
    if ( *(_DWORD *)(a2 + 40) <= v8 )
      return v10;
    v14 = v8;
    while ( 1 )
    {
      v16 = a3;
      if ( v23 >> 4 > 0 )
      {
        while ( *(_DWORD *)v16 != v14 )
        {
          if ( *((_DWORD *)v16 + 1) == v14 )
          {
            v16 += 4;
            goto LABEL_22;
          }
          if ( *((_DWORD *)v16 + 2) == v14 )
          {
            v16 += 8;
            goto LABEL_22;
          }
          if ( *((_DWORD *)v16 + 3) == v14 )
          {
            v16 += 12;
            goto LABEL_22;
          }
          v16 += 16;
          if ( v16 == &a3[v23 & 0xFFFFFFFFFFFFFFF0LL] )
            goto LABEL_27;
        }
        goto LABEL_22;
      }
LABEL_27:
      v17 = v21 - v16;
      if ( v21 - v16 == 8 )
        goto LABEL_37;
      if ( v17 != 12 )
      {
        if ( v17 != 4 )
          goto LABEL_31;
        goto LABEL_30;
      }
      if ( *(_DWORD *)v16 != v14 )
        break;
LABEL_22:
      if ( v16 != v21 )
      {
        v15 = (_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v14);
        if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *, unsigned __int64, _QWORD, unsigned int *, unsigned int *, __int64))(*a6 + 120LL))(
                a6,
                *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 24LL) + 16LL * (v15[2] & 0x7FFFFFFF))
              & 0xFFFFFFFFFFFFFFF8LL,
                (*v15 >> 8) & 0xFFF,
                &v24,
                &v25,
                a1) )
          sub_16BD130("cannot spill patchpoint subregister operand", 1u);
        v26.m128i_i64[0] = 1;
        v27 = 0;
        v28 = 1;
        sub_1E1A9C0((__int64)v10, a1, &v26);
        v26.m128i_i64[0] = 1;
        v28 = v24;
        v27 = 0;
        sub_1E1A9C0((__int64)v10, a1, &v26);
        v26.m128i_i64[0] = 5;
        v27 = 0;
        LODWORD(v28) = a5;
        sub_1E1A9C0((__int64)v10, a1, &v26);
        v26.m128i_i64[0] = 1;
        v27 = 0;
        v28 = v25;
        sub_1E1A9C0((__int64)v10, a1, &v26);
        goto LABEL_25;
      }
LABEL_31:
      sub_1E1A9C0((__int64)v10, a1, (const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL * v14));
LABEL_25:
      if ( *(_DWORD *)(a2 + 40) <= ++v14 )
        return v10;
    }
    v16 += 4;
LABEL_37:
    if ( *(_DWORD *)v16 != v14 )
    {
      v16 += 4;
LABEL_30:
      if ( *(_DWORD *)v16 != v14 )
        goto LABEL_31;
      goto LABEL_22;
    }
    goto LABEL_22;
  }
  v8 = 2;
LABEL_5:
  v9 = (unsigned int *)a3;
  while ( *v9 >= v8 )
  {
    if ( v21 == (char *)++v9 )
      goto LABEL_10;
  }
  return 0;
}
