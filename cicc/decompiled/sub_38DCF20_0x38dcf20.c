// Function: sub_38DCF20
// Address: 0x38dcf20
//
void __fastcall sub_38DCF20(__int64 a1, __int64 a2)
{
  char v3; // r14
  char v4; // si
  char *v5; // rax
  char v6; // al
  void (*v7)(); // rcx
  _QWORD v8[2]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned __int64 v9; // [rsp+10h] [rbp-E0h]
  char *v10; // [rsp+18h] [rbp-D8h]
  int v11; // [rsp+20h] [rbp-D0h]
  unsigned __int64 *v12; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v13[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v14[176]; // [rsp+40h] [rbp-B0h] BYREF

  v13[1] = 0x8000000000LL;
  v13[0] = (unsigned __int64)v14;
  v11 = 1;
  v8[0] = &unk_49EFC48;
  v10 = 0;
  v9 = 0;
  v8[1] = 0;
  v12 = v13;
  sub_16E7A40((__int64)v8, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      v6 = a2;
      v4 = a2 & 0x7F;
      a2 >>= 7;
      if ( !a2 )
      {
        v3 = 0;
        if ( (v6 & 0x40) == 0 )
          goto LABEL_4;
        goto LABEL_3;
      }
      if ( a2 == -1 )
      {
        v3 = 0;
        if ( (v6 & 0x40) != 0 )
          break;
      }
LABEL_3:
      v4 |= 0x80u;
      v3 = 1;
LABEL_4:
      v5 = v10;
      if ( (unsigned __int64)v10 >= v9 )
        goto LABEL_10;
LABEL_5:
      v10 = v5 + 1;
      *v5 = v4;
      if ( !v3 )
        goto LABEL_11;
    }
    v5 = v10;
    if ( (unsigned __int64)v10 < v9 )
      goto LABEL_5;
LABEL_10:
    sub_16E7DE0((__int64)v8, v4);
  }
  while ( v3 );
LABEL_11:
  v7 = *(void (**)())(*(_QWORD *)a1 + 400LL);
  if ( v7 != nullsub_1953 )
    ((void (__fastcall *)(__int64, unsigned __int64, _QWORD))v7)(a1, *v12, *((unsigned int *)v12 + 2));
  v8[0] = &unk_49EFD28;
  sub_16E7960((__int64)v8);
  if ( (_BYTE *)v13[0] != v14 )
    _libc_free(v13[0]);
}
