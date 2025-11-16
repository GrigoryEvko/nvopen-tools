// Function: sub_A830B0
// Address: 0xa830b0
//
__int64 __fastcall sub_A830B0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v7; // r15
  unsigned int v8; // r14d
  unsigned int v9; // eax
  unsigned int *v11; // rdi
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  unsigned int *v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rsi
  unsigned int *v20; // rdi
  __int64 (__fastcall *v21)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v22; // r13
  unsigned int *v23; // rbx
  unsigned int *v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rsi
  _BYTE v28[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v29; // [rsp+30h] [rbp-40h]

  v5 = a2;
  v7 = *(_QWORD *)(a2 + 8);
  v8 = sub_BCB060(v7);
  v9 = sub_BCB060(a3);
  if ( v8 >= v9 )
  {
    if ( v8 == v9 || v7 == a3 )
      return v5;
    v20 = a1[10];
    v21 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v20 + 120LL);
    if ( v21 == sub_920130 )
    {
      if ( *(_BYTE *)a2 > 0x15u )
      {
LABEL_24:
        v29 = 257;
        v5 = sub_B51D30(38, a2, a3, v28, 0, 0);
        (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11]
                                                                                                 + 16LL))(
          a1[11],
          v5,
          a4,
          a1[7],
          a1[8]);
        v22 = 4LL * *((unsigned int *)a1 + 2);
        v23 = *a1;
        v24 = &v23[v22];
        while ( v24 != v23 )
        {
          v25 = *((_QWORD *)v23 + 1);
          v26 = *v23;
          v23 += 4;
          sub_B99FD0(v5, v26, v25);
        }
        return v5;
      }
      if ( (unsigned __int8)sub_AC4810(38) )
        v13 = sub_ADAB70(38, a2, a3, 0);
      else
        v13 = sub_AA93C0(38, a2, a3);
    }
    else
    {
      v13 = v21((__int64)v20, 38u, (_BYTE *)a2, a3);
    }
    if ( v13 )
      return v13;
    goto LABEL_24;
  }
  if ( v7 == a3 )
    return v5;
  v11 = a1[10];
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v11 + 120LL);
  if ( v12 != sub_920130 )
  {
    v13 = v12((__int64)v11, 39u, (_BYTE *)a2, a3);
    goto LABEL_10;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(39) )
      v13 = sub_ADAB70(39, a2, a3, 0);
    else
      v13 = sub_AA93C0(39, a2, a3);
LABEL_10:
    if ( v13 )
      return v13;
  }
  v29 = 257;
  v14 = sub_BD2C40(72, unk_3F10A14);
  v15 = v14;
  if ( v14 )
    sub_B515B0(v14, a2, a3, v28, 0, 0);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v15,
    a4,
    a1[7],
    a1[8]);
  v16 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v16 )
  {
    v17 = *a1;
    do
    {
      v18 = *((_QWORD *)v17 + 1);
      v19 = *v17;
      v17 += 4;
      sub_B99FD0(v15, v19, v18);
    }
    while ( (unsigned int *)v16 != v17 );
  }
  return v15;
}
