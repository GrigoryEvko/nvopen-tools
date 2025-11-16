// Function: sub_3154400
// Address: 0x3154400
//
__int64 __fastcall sub_3154400(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v20; // [rsp+10h] [rbp-80h]
  __int64 v21; // [rsp+18h] [rbp-78h] BYREF
  char v22; // [rsp+2Fh] [rbp-61h] BYREF
  __int64 v23; // [rsp+30h] [rbp-60h] BYREF
  __int64 v24; // [rsp+38h] [rbp-58h] BYREF
  __int64 v25; // [rsp+40h] [rbp-50h] BYREF
  char v26; // [rsp+49h] [rbp-47h]
  _QWORD *v27; // [rsp+50h] [rbp-40h]
  char v28; // [rsp+59h] [rbp-37h]

  v6 = *(_QWORD *)a1;
  v21 = a2;
  (*(void (__fastcall **)(__int64))(v6 + 104))(a1);
  v7 = *(_QWORD *)a1;
  v22 = 0;
  v23 = 0;
  (*(void (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(v7 + 120))(
    a1,
    "Guid",
    1,
    0,
    &v22,
    &v23);
  sub_3153850(a1, &v21);
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, 0);
  (*(void (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
    a1,
    "Counters",
    1,
    0,
    &v22,
    &v23);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
  v20 = *(unsigned int *)(a3 + 8);
  if ( *(_DWORD *)(a3 + 8) )
  {
    v8 = 0;
    do
    {
      (*(void (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 72LL))(a1, (unsigned int)v8, &v23);
      v9 = *(_QWORD *)(*(_QWORD *)a3 + 8 * v8++);
      v25 = v9;
      sub_3153850(a1, &v25);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 80LL))(a1, v23);
    }
    while ( v20 != v8 );
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, 0);
  if ( a4[5] )
  {
    v10 = 0;
    (*(void (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
      a1,
      "Callsites",
      1,
      0,
      &v22,
      &v23);
    v11 = a4[3];
    v26 = 1;
    v27 = a4 + 1;
    v25 = v11;
    v28 = 1;
    v12 = sub_3154370(&v25);
    v24 = 0;
    v13 = v12;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
    do
    {
      (*(void (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, v10, &v24);
      v14 = a4[2];
      if ( !v14 )
        goto LABEL_13;
      v15 = a4 + 1;
      do
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(v14 + 16);
          v17 = *(_QWORD *)(v14 + 24);
          if ( v10 <= *(_DWORD *)(v14 + 32) )
            break;
          v14 = *(_QWORD *)(v14 + 24);
          if ( !v17 )
            goto LABEL_11;
        }
        v15 = (_QWORD *)v14;
        v14 = *(_QWORD *)(v14 + 16);
      }
      while ( v16 );
LABEL_11:
      if ( a4 + 1 != v15 && v10 >= *((_DWORD *)v15 + 8) )
      {
        sub_3154660(a1, v15 + 5, v17, v16);
      }
      else
      {
LABEL_13:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
      }
      ++v10;
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 40LL))(a1, 0);
    }
    while ( *(_DWORD *)(v13 + 32) >= v10 );
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, 0);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
