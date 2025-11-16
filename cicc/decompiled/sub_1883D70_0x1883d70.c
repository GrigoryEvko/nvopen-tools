// Function: sub_1883D70
// Address: 0x1883d70
//
__int64 __fastcall sub_1883D70(__int64 a1, __int64 *a2)
{
  int v3; // ebx
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  char *v10; // rsi
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // r8
  _QWORD *v14; // r9
  __int64 v15; // rcx
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdi
  unsigned __int64 v20; // [rsp+8h] [rbp-78h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  _QWORD *v23; // [rsp+20h] [rbp-60h]
  __int64 v24; // [rsp+28h] [rbp-58h]
  char v25; // [rsp+3Fh] [rbp-41h] BYREF
  __int64 v26; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v27[7]; // [rsp+48h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -858993459 * ((a2[1] - *a2) >> 3);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 1;
    v6 = 0;
    v24 = v4 + 2;
    do
    {
      while ( 1 )
      {
        v7 = v6 + 40;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v5 - 1),
               &v26) )
        {
          break;
        }
        v6 += 40;
        if ( v24 == ++v5 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v8 = *a2;
      v9 = 0xCCCCCCCCCCCCCCCDLL * ((a2[1] - *a2) >> 3);
      if ( v9 <= v5 - 1 )
      {
        if ( v9 < v5 )
        {
          sub_1883B00((__int64)a2, v5 - v9);
          v8 = *a2;
        }
        else if ( v9 > v5 && a2[1] != v8 + v7 )
        {
          v22 = v6;
          v17 = a2[1];
          v20 = v5;
          v18 = v8 + v7;
          do
          {
            v19 = *(_QWORD *)(v18 + 16);
            if ( v19 )
              j_j___libc_free_0(v19, *(_QWORD *)(v18 + 32) - v19);
            v18 += 40;
          }
          while ( v17 != v18 );
          v6 = v22;
          v5 = v20;
          a2[1] = v8 + v7;
          v8 = *a2;
        }
      }
      v23 = (_QWORD *)(v8 + v6);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      v10 = "VFunc";
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "VFunc",
             0,
             0,
             &v25) )
      {
        sub_187A340(a1, v23);
        v10 = (char *)v27[0];
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v27[0]);
      }
      v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
      v14 = v27;
      if ( !v11 || (v15 = v23[2], v23[3] != v15) )
      {
        v10 = "Args";
        if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
               a1,
               "Args",
               0,
               0,
               &v25,
               v27) )
        {
          sub_1883700(a1, v23 + 2);
          v10 = (char *)v27[0];
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v27[0]);
        }
      }
      v6 = v7;
      ++v5;
      (*(void (__fastcall **)(__int64, char *, __int64, __int64, __int64, _QWORD *))(*(_QWORD *)a1 + 112LL))(
        a1,
        v10,
        v12,
        v15,
        v13,
        v14);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v26);
    }
    while ( v24 != v5 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
