// Function: sub_E98EB0
// Address: 0xe98eb0
//
__int64 __fastcall sub_E98EB0(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned int v4; // r12d
  char v6; // si
  char *v7; // rax
  unsigned int v8; // r13d
  char *v9; // rax
  void (*v10)(); // rcx
  __int64 v11; // rsi
  unsigned int v12; // ebx
  char *v14; // rax
  _QWORD v15[3]; // [rsp+10h] [rbp-110h] BYREF
  unsigned __int64 v16; // [rsp+28h] [rbp-F8h]
  char *v17; // [rsp+30h] [rbp-F0h]
  __int64 v18; // [rsp+38h] [rbp-E8h]
  __int64 *v19; // [rsp+40h] [rbp-E0h]
  _BYTE *v20; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v21; // [rsp+58h] [rbp-C8h]
  __int64 v22; // [rsp+60h] [rbp-C0h]
  _BYTE v23[184]; // [rsp+68h] [rbp-B8h] BYREF

  v4 = 0;
  v20 = v23;
  v18 = 0x100000000LL;
  v15[0] = &unk_49DD288;
  v21 = 0;
  v22 = 128;
  v15[1] = 2;
  v15[2] = 0;
  v16 = 0;
  v17 = 0;
  v19 = (__int64 *)&v20;
  sub_CB5980((__int64)v15, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      ++v4;
      v6 = a2 & 0x7F;
      a2 >>= 7;
      if ( a2 || a3 > v4 )
        v6 |= 0x80u;
      v7 = v17;
      if ( (unsigned __int64)v17 >= v16 )
        break;
      ++v17;
      *v7 = v6;
      if ( !a2 )
        goto LABEL_7;
    }
    sub_CB5D20((__int64)v15, v6);
  }
  while ( a2 );
LABEL_7:
  if ( a3 > v4 )
  {
    v8 = a3 - 1;
    if ( v4 < v8 )
    {
      do
      {
        v14 = v17;
        if ( (unsigned __int64)v17 < v16 )
        {
          ++v17;
          *v14 = 0x80;
        }
        else
        {
          sub_CB5D20((__int64)v15, 128);
        }
        ++v4;
      }
      while ( v8 != v4 );
    }
    v9 = v17;
    if ( (unsigned __int64)v17 >= v16 )
    {
      sub_CB5D20((__int64)v15, 0);
    }
    else
    {
      ++v17;
      *v9 = 0;
    }
  }
  v10 = *(void (**)())(*(_QWORD *)a1 + 512LL);
  v11 = *v19;
  if ( v10 != nullsub_360 )
    ((void (__fastcall *)(__int64, __int64, __int64))v10)(a1, v11, v19[1]);
  v12 = v21;
  v15[0] = &unk_49DD388;
  sub_CB5840((__int64)v15);
  if ( v20 != v23 )
    _libc_free(v20, v11);
  return v12;
}
