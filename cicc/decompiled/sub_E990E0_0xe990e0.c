// Function: sub_E990E0
// Address: 0xe990e0
//
__int64 __fastcall sub_E990E0(__int64 a1, __int64 a2)
{
  char v3; // r14
  char v4; // si
  char *v5; // rax
  char v6; // al
  void (*v7)(); // rcx
  __int64 v8; // rsi
  unsigned int v9; // ebx
  _QWORD v11[3]; // [rsp+0h] [rbp-110h] BYREF
  unsigned __int64 v12; // [rsp+18h] [rbp-F8h]
  char *v13; // [rsp+20h] [rbp-F0h]
  __int64 v14; // [rsp+28h] [rbp-E8h]
  __int64 *v15; // [rsp+30h] [rbp-E0h]
  _BYTE *v16; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v17; // [rsp+48h] [rbp-C8h]
  __int64 v18; // [rsp+50h] [rbp-C0h]
  _BYTE v19[184]; // [rsp+58h] [rbp-B8h] BYREF

  v14 = 0x100000000LL;
  v16 = v19;
  v17 = 0;
  v11[0] = &unk_49DD288;
  v18 = 128;
  v11[1] = 2;
  v11[2] = 0;
  v12 = 0;
  v13 = 0;
  v15 = (__int64 *)&v16;
  sub_CB5980((__int64)v11, 0, 0, 0);
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
      v5 = v13;
      if ( (unsigned __int64)v13 >= v12 )
        goto LABEL_10;
LABEL_5:
      v13 = v5 + 1;
      *v5 = v4;
      if ( !v3 )
        goto LABEL_11;
    }
    v5 = v13;
    if ( (unsigned __int64)v13 < v12 )
      goto LABEL_5;
LABEL_10:
    sub_CB5D20((__int64)v11, v4);
  }
  while ( v3 );
LABEL_11:
  v7 = *(void (**)())(*(_QWORD *)a1 + 512LL);
  v8 = *v15;
  if ( v7 != nullsub_360 )
    ((void (__fastcall *)(__int64, __int64, __int64))v7)(a1, v8, v15[1]);
  v9 = v17;
  v11[0] = &unk_49DD388;
  sub_CB5840((__int64)v11);
  if ( v16 != v19 )
    _libc_free(v16, v8);
  return v9;
}
