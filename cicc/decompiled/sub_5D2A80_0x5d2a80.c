// Function: sub_5D2A80
// Address: 0x5d2a80
//
__int64 __fastcall sub_5D2A80(
        unsigned int a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 (__fastcall *a5)(_QWORD, _QWORD),
        __int64 (*a6)(void),
        void (*a7)(void),
        __int64 a8)
{
  _BOOL8 v9; // rbx
  char *v10; // rdi
  int v11; // r12d
  rlim_t rlim_cur; // [rsp+8h] [rbp-C8h]
  _BYTE v17[16]; // [rsp+40h] [rbp-90h] BYREF
  _BYTE v18[16]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v19[16]; // [rsp+60h] [rbp-70h] BYREF
  _BYTE v20[16]; // [rsp+70h] [rbp-60h] BYREF
  _BYTE v21[16]; // [rsp+80h] [rbp-50h] BYREF
  _BYTE v22[16]; // [rsp+90h] [rbp-40h] BYREF
  struct rlimit rlimits; // [rsp+A0h] [rbp-30h] BYREF

  if ( !_setjmp(env) )
  {
    LODWORD(v9) = 0;
    *(_QWORD *)a8 = 0;
    unk_4F07510 = stderr;
    sub_705D80();
    unk_4D045D8 = a5;
    unk_4D045D0 = a6;
    sub_721180(v17);
    sub_617BD0(a1, a2);
    if ( unk_4D04704 && !getrlimit(RLIMIT_STACK, &rlimits) )
    {
      rlim_cur = rlimits.rlim_cur;
      rlimits.rlim_cur = rlimits.rlim_max;
      v9 = setrlimit(RLIMIT_STACK, &rlimits) == 0;
    }
    sub_705DD0();
    if ( unk_4D04744 )
      sub_721180(v18);
    sub_8D0F00();
    v10 = qword_4F076F0;
    sub_8D0BC0(qword_4F076F0);
    sub_709330(v10, 1);
    if ( unk_4D04744 )
    {
      sub_721180(v19);
      sub_7211D0("Front end time", v18, v19);
    }
    if ( unk_4F074B0 )
    {
      unk_4D048FC = 1;
    }
    else if ( !unk_4D048FC )
    {
      if ( unk_4D04744 )
        sub_721180(v20);
      sub_5E3AD0(a3, a4);
      if ( unk_4D04744 )
      {
        sub_721180(v21);
        sub_7211D0("Back end time", v20, v21);
      }
    }
    LOBYTE(v11) = 8;
    unk_4D045D8("Front End Cleanup", byte_3F871B3);
    sub_823310(0);
    sub_709700();
    unk_4D045D0();
    if ( !unk_4F074B0 )
      v11 = unk_4F074A8 == 0 ? 3 : 5;
    if ( unk_4D04744 )
    {
      sub_721180(v22);
      sub_7211D0("Total compilation time", v17, v22);
    }
    if ( v9 )
    {
      rlimits.rlim_cur = rlim_cur;
      setrlimit(RLIMIT_STACK, &rlimits);
    }
    sub_720FF0((unsigned __int8)v11);
  }
  unk_4D045D8("Front End Cleanup", byte_3F871B3);
  sub_709710();
  unk_4D045D0();
  if ( !unk_4F07668 )
    return *(_QWORD *)a8;
  a7();
  return 0;
}
