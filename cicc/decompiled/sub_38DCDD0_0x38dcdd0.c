// Function: sub_38DCDD0
// Address: 0x38dcdd0
//
void __fastcall sub_38DCDD0(__int64 a1, unsigned __int64 a2)
{
  char v3; // si
  char v4; // al
  char *v5; // rax
  void (*v6)(); // rcx
  _QWORD v7[2]; // [rsp+0h] [rbp-E0h] BYREF
  unsigned __int64 v8; // [rsp+10h] [rbp-D0h]
  char *v9; // [rsp+18h] [rbp-C8h]
  int v10; // [rsp+20h] [rbp-C0h]
  unsigned __int64 *v11; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v12[2]; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE v13[160]; // [rsp+40h] [rbp-A0h] BYREF

  v12[1] = 0x8000000000LL;
  v12[0] = (unsigned __int64)v13;
  v10 = 1;
  v7[0] = &unk_49EFC48;
  v9 = 0;
  v8 = 0;
  v7[1] = 0;
  v11 = v12;
  sub_16E7A40((__int64)v7, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      v3 = a2 & 0x7F;
      v4 = a2 & 0x7F | 0x80;
      a2 >>= 7;
      if ( a2 )
        v3 = v4;
      v5 = v9;
      if ( (unsigned __int64)v9 >= v8 )
        break;
      ++v9;
      *v5 = v3;
      if ( !a2 )
        goto LABEL_7;
    }
    sub_16E7DE0((__int64)v7, v3);
  }
  while ( a2 );
LABEL_7:
  v6 = *(void (**)())(*(_QWORD *)a1 + 400LL);
  if ( v6 != nullsub_1953 )
    ((void (__fastcall *)(__int64, unsigned __int64, _QWORD))v6)(a1, *v11, *((unsigned int *)v11 + 2));
  v7[0] = &unk_49EFD28;
  sub_16E7960((__int64)v7);
  if ( (_BYTE *)v12[0] != v13 )
    _libc_free(v12[0]);
}
