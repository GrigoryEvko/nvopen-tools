// Function: sub_E761F0
// Address: 0xe761f0
//
__int64 __fastcall sub_E761F0(_QWORD *a1, __int64 a2, int **a3)
{
  __int64 result; // rax
  char v4; // r12
  int v5; // r15d
  __int64 v6; // r13
  int *v7; // r14
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned int v12; // eax
  char v13; // al
  unsigned __int64 v14; // rax
  char v15; // al
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r15
  char v20; // [rsp+Bh] [rbp-65h]
  int v21; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v22; // [rsp+Ch] [rbp-64h]
  int v23; // [rsp+Ch] [rbp-64h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  int v25; // [rsp+18h] [rbp-58h]
  unsigned int v26; // [rsp+1Ch] [rbp-54h]
  unsigned int v27; // [rsp+20h] [rbp-50h]
  __int64 v28; // [rsp+20h] [rbp-50h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  char v30; // [rsp+30h] [rbp-40h]
  int v31; // [rsp+34h] [rbp-3Ch]
  char v32; // [rsp+34h] [rbp-3Ch]
  unsigned int v33; // [rsp+38h] [rbp-38h]
  __int64 v34; // [rsp+38h] [rbp-38h]

  result = (__int64)a3[1];
  v29 = result;
  if ( *a3 == (int *)result )
    return result;
  v20 = 0;
  v4 = 1;
  v5 = 0;
  v6 = 0;
  v7 = *a3;
  v31 = 0;
  v30 = 1;
  result = 1;
  LODWORD(v8) = 1;
  do
  {
    while ( 1 )
    {
      v11 = *((_QWORD *)v7 + 3);
      if ( v11 )
      {
        if ( !v4 )
        {
          v9 = v6;
          v10 = v6;
          v6 = 0;
          v5 = 0;
          (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*a1 + 1304LL))(a1, a2, v10, v9);
          v11 = *((_QWORD *)v7 + 3);
          LODWORD(result) = 1;
          v31 = 0;
          v30 = 1;
          LODWORD(v8) = 1;
        }
        v27 = result;
        v4 = 1;
        v33 = v8;
        (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 208LL))(a1, v11, *((_QWORD *)v7 + 4));
        v8 = v33;
        result = v27;
        goto LABEL_6;
      }
      v4 = *((_BYTE *)v7 + 40);
      v34 = *((_QWORD *)v7 + 2);
      v28 = *(_QWORD *)(a1[1] + 152LL);
      if ( !v4 )
        break;
      v17 = v6;
      v6 = 0;
      v5 = 0;
      (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD))(*a1 + 1312LL))(
        a1,
        0x7FFFFFFFFFFFFFFFLL,
        v17,
        v34,
        *(unsigned int *)(*(_QWORD *)(a1[1] + 152LL) + 8LL));
      v20 = v4;
      result = 1;
      v8 = 1;
      v31 = 0;
      v30 = 1;
LABEL_6:
      v7 += 12;
      if ( (int *)v29 == v7 )
        goto LABEL_28;
    }
    v24 = (unsigned int)v7[1] - result;
    v26 = *v7;
    if ( (_DWORD)v8 != *v7 )
    {
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 4, 1);
      sub_E98EB0(a1, v26, 0);
    }
    v21 = *((unsigned __int16 *)v7 + 4);
    v25 = v21;
    if ( v21 != v5 )
    {
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 5, 1);
      sub_E98EB0(a1, (unsigned __int16)v21, 0);
    }
    v12 = v7[3];
    if ( v12 && *(_WORD *)(a1[1] + 1904LL) > 3u )
    {
      v18 = v12;
      v23 = sub_F03EF0(v12);
      (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 536LL))(a1, 0, 1);
      sub_E98EB0(a1, (unsigned int)(v23 + 1), 0);
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 4, 1);
      sub_E98EB0(a1, v18, 0);
    }
    v22 = *((_BYTE *)v7 + 11);
    if ( v22 != v31 )
    {
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 12, 1);
      sub_E98EB0(a1, v22, 0);
    }
    v32 = *((_BYTE *)v7 + 10);
    v13 = v32;
    if ( (((unsigned __int8)v32 ^ (unsigned __int8)v30) & 1) != 0 )
    {
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 6, 1);
      v13 = *((_BYTE *)v7 + 10);
      v30 = v32;
    }
    if ( (v13 & 2) != 0 )
      (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 7, 1);
    v14 = *(unsigned int *)(a1[1] + 56LL);
    if ( (unsigned int)v14 > 0x3A || (v16 = 0x4000C0000200000LL, !_bittest64(&v16, v14)) )
    {
      v15 = *((_BYTE *)v7 + 10);
      if ( (v15 & 4) != 0 )
      {
        (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 10, 1);
        v15 = *((_BYTE *)v7 + 10);
      }
      if ( (v15 & 8) != 0 )
        (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 11, 1);
    }
    v7 += 12;
    (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, _QWORD))(*a1 + 1312LL))(
      a1,
      v24,
      v6,
      v34,
      *(unsigned int *)(v28 + 8));
    v31 = v22;
    result = (unsigned int)*(v7 - 11);
    v6 = v34;
    v5 = v25;
    v8 = v26;
  }
  while ( (int *)v29 != v7 );
LABEL_28:
  if ( !v20 && !v4 )
    return (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int64))(*a1 + 1304LL))(a1, a2, v6, 0, v8);
  return result;
}
