// Function: sub_2F1E8D0
// Address: 0x2f1e8d0
//
__int64 __fastcall sub_2F1E8D0(__int64 a1, __int64 *a2)
{
  int v2; // ebx
  __int64 v3; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  unsigned int *v11; // [rsp+8h] [rbp-68h]
  __int64 v12; // [rsp+18h] [rbp-58h]
  char v13; // [rsp+2Fh] [rbp-41h] BYREF
  __int64 v14; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v15[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v2 = -858993459 * ((a2[1] - *a2) >> 2);
  if ( v2 )
  {
    v3 = (unsigned int)(v2 - 1);
    v4 = 1;
    v5 = 0;
    v12 = v3 + 2;
    do
    {
      while ( 1 )
      {
        v6 = v5 + 20;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
               a1,
               (unsigned int)(v4 - 1),
               &v14) )
        {
          break;
        }
        v5 += 20;
        if ( v12 == ++v4 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v7 = *a2;
      v8 = 0xCCCCCCCCCCCCCCCDLL * ((a2[1] - *a2) >> 2);
      if ( v8 <= v4 - 1 )
      {
        if ( v8 < v4 )
        {
          sub_2F1E6D0((__int64)a2, v4 - v8);
          v7 = *a2;
        }
        else if ( v8 > v4 && a2[1] != v7 + v6 )
        {
          a2[1] = v7 + v6;
        }
      }
      v11 = (unsigned int *)(v7 + v5);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 144LL))(a1);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "srcinst",
             1,
             0,
             &v13,
             v15) )
      {
        sub_2F07DB0(a1, v11);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "srcop",
             1,
             0,
             &v13,
             v15) )
      {
        sub_2F07DB0(a1, v11 + 1);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "dstinst",
             1,
             0,
             &v13,
             v15) )
      {
        sub_2F07DB0(a1, v11 + 2);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "dstop",
             1,
             0,
             &v13,
             v15) )
      {
        sub_2F07DB0(a1, v11 + 3);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "subreg",
             1,
             0,
             &v13,
             v15) )
      {
        sub_2F07DB0(a1, v11 + 4);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
      }
      v5 += 20;
      ++v4;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 152LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v14);
    }
    while ( v12 != v4 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
