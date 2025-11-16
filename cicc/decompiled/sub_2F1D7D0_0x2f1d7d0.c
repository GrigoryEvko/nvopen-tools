// Function: sub_2F1D7D0
// Address: 0x2f1d7d0
//
__int64 __fastcall sub_2F1D7D0(__int64 a1, unsigned __int64 *a2)
{
  __int64 v2; // r15
  int v3; // ebx
  __int64 v4; // rax
  __int64 v5; // r12
  unsigned __int64 v6; // rbx
  __int64 v8; // r13
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 *v12; // r15
  unsigned __int64 *v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+28h] [rbp-58h]
  char v21; // [rsp+3Fh] [rbp-41h] BYREF
  __int64 v22; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v23[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a1;
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = -1431655765 * ((__int64)(a2[1] - *a2) >> 6);
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = 1;
    v20 = v4 + 2;
    do
    {
      while ( 1 )
      {
        v8 = v5 + 192;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v6 - 1),
               &v22) )
        {
          break;
        }
        v5 += 192;
        if ( v20 == ++v6 )
          goto LABEL_17;
      }
      v9 = *a2;
      v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(a2[1] - *a2) >> 6);
      if ( v10 <= v6 - 1 )
      {
        if ( v10 < v6 )
        {
          sub_2F1D340(a2, v6 - v10);
          v9 = *a2;
        }
        else if ( v10 > v6 && a2[1] != v9 + v8 )
        {
          v17 = v6;
          v12 = (unsigned __int64 *)a2[1];
          v13 = (unsigned __int64 *)(v9 + v8);
          do
          {
            v14 = v13[18];
            if ( (unsigned __int64 *)v14 != v13 + 20 )
              j_j___libc_free_0(v14);
            v15 = v13[12];
            if ( (unsigned __int64 *)v15 != v13 + 14 )
              j_j___libc_free_0(v15);
            v16 = v13[6];
            if ( (unsigned __int64 *)v16 != v13 + 8 )
              j_j___libc_free_0(v16);
            if ( (unsigned __int64 *)*v13 != v13 + 2 )
              j_j___libc_free_0(*v13);
            v13 += 24;
          }
          while ( v12 != v13 );
          v6 = v17;
          a2[1] = v9 + v8;
          v9 = *a2;
        }
      }
      v19 = v9 + v5;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 144LL))(a1);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "entry-value-register",
             1,
             0,
             &v21,
             v23) )
      {
        sub_2F0E9C0(a1, v19);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v23[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "debug-info-variable",
             1,
             0,
             &v21,
             v23) )
      {
        sub_2F0E9C0(a1, v19 + 48);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v23[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "debug-info-expression",
             1,
             0,
             &v21,
             v23) )
      {
        sub_2F0E9C0(a1, v19 + 96);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v23[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "debug-info-location",
             1,
             0,
             &v21,
             v23) )
      {
        sub_2F0E9C0(a1, v19 + 144);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v23[0]);
      }
      v5 += 192;
      ++v6;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 152LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v22);
    }
    while ( v20 != v6 );
LABEL_17:
    v2 = a1;
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 48LL))(v2);
}
