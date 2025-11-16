// Function: sub_151AEF0
// Address: 0x151aef0
//
__int64 *__fastcall sub_151AEF0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v6[2]; // [rsp+0h] [rbp-250h] BYREF
  char v7; // [rsp+10h] [rbp-240h]
  char v8; // [rsp+11h] [rbp-23Fh]
  const char *v9; // [rsp+20h] [rbp-230h] BYREF
  __int64 v10; // [rsp+28h] [rbp-228h]
  _BYTE v11[544]; // [rsp+30h] [rbp-220h] BYREF

  if ( sub_15127D0(*(_QWORD *)(a2 + 232), 22, 0) )
  {
    v11[1] = 1;
    v9 = "Invalid record";
    v11[0] = 3;
    sub_1514BE0(a1, (__int64)&v9);
  }
  else
  {
    v9 = v11;
    v10 = 0x4000000000LL;
    while ( 1 )
    {
      v3 = sub_14ED070(*(_QWORD *)(a2 + 232), 0);
      if ( (_DWORD)v3 == 1 )
        break;
      if ( (v3 & 0xFFFFFFFD) == 0 )
      {
        v8 = 1;
        v6[0] = (__int64)"Malformed block";
        v7 = 3;
        sub_1514BE0(a1, (__int64)v6);
        goto LABEL_11;
      }
      v4 = *(_QWORD *)(a2 + 232);
      LODWORD(v10) = 0;
      if ( (unsigned int)sub_1510D70(v4, SHIDWORD(v3), (__int64)&v9, 0) == 6 )
      {
        sub_151ABA0(v6, a2, (__int64 **)&v9);
        if ( (v6[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        {
          *a1 = v6[0] & 0xFFFFFFFFFFFFFFFELL | 1;
          goto LABEL_11;
        }
      }
    }
    *a1 = 1;
LABEL_11:
    if ( v9 != v11 )
      _libc_free((unsigned __int64)v9);
  }
  return a1;
}
