// Function: sub_1B28080
// Address: 0x1b28080
//
__int64 __fastcall sub_1B28080(__int64 a1)
{
  void *v3; // rax
  _QWORD *v4; // rdi
  _BYTE *v5; // rax
  __int64 v6; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-60h] BYREF
  char v8; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v9[4]; // [rsp+20h] [rbp-40h] BYREF
  int v10; // [rsp+40h] [rbp-20h]
  _QWORD *v11; // [rsp+48h] [rbp-18h]

  if ( *(_BYTE *)(a1 + 16) )
  {
    v3 = sub_16E8CB0();
    sub_155C2B0(a1, (__int64)v3, 0);
    v4 = sub_16E8CB0();
    v5 = (_BYTE *)v4[3];
    if ( (unsigned __int64)v5 >= v4[2] )
    {
      sub_16E7DE0((__int64)v4, 10);
    }
    else
    {
      v4[3] = v5 + 1;
      *v5 = 10;
    }
    v8 = 0;
    v7[0] = &v8;
    v7[1] = 0;
    v10 = 1;
    memset(&v9[1], 0, 24);
    v9[0] = &unk_49EFBE0;
    v11 = v7;
    v6 = sub_16E7EE0((__int64)v9, "Sanitizer interface function redefined: ", 0x28u);
    sub_155C2B0(a1, v6, 0);
    sub_16BD160((__int64)v7, 1u);
  }
  return a1;
}
