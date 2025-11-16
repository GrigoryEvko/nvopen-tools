// Function: sub_3117620
// Address: 0x3117620
//
__int64 __fastcall sub_3117620(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  unsigned __int64 v5; // r15
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned int *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-58h]
  char v13; // [rsp+1Fh] [rbp-41h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v15[7]; // [rsp+28h] [rbp-38h] BYREF

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int64 *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Hash",
         1,
         0,
         &v14,
         v15) )
  {
    sub_3114D70(a1, (__int64 *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, __int64 *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Terminals",
         1,
         0,
         &v14,
         v15) )
  {
    sub_3115110(a1, (unsigned int *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v15[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SuccessorIds",
         1,
         0,
         &v13,
         &v14) )
  {
    LODWORD(v4) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
      v4 = (__int64)(*(_QWORD *)(a2 + 24) - *(_QWORD *)(a2 + 16)) >> 2;
    if ( (_DWORD)v4 )
    {
      v5 = 0;
      v6 = (unsigned int)(v4 - 1) + 2LL;
      v7 = 1;
      v12 = v6;
      do
      {
        while ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD *))(*(_QWORD *)a1 + 72LL))(
                   a1,
                   (unsigned int)v5,
                   v15) )
        {
          ++v5;
          if ( v12 == ++v7 )
            goto LABEL_15;
        }
        v8 = *(_QWORD *)(a2 + 16);
        v9 = (*(_QWORD *)(a2 + 24) - v8) >> 2;
        if ( v5 >= v9 )
        {
          if ( v9 < v7 )
          {
            sub_C17A60(a2 + 16, v7 - v9);
            v8 = *(_QWORD *)(a2 + 16);
          }
          else if ( v9 > v7 )
          {
            v11 = v8 + 4 * v7;
            if ( *(_QWORD *)(a2 + 24) != v11 )
              *(_QWORD *)(a2 + 24) = v11;
          }
        }
        v10 = (unsigned int *)(v8 + 4 * v5++);
        ++v7;
        sub_3115110(a1, v10);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 80LL))(a1, v15[0]);
      }
      while ( v12 != v7 );
    }
LABEL_15:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v14);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
