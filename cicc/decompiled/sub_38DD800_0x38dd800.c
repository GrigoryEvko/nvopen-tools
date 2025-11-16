// Function: sub_38DD800
// Address: 0x38dd800
//
void __fastcall sub_38DD800(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  char *v3; // r14
  unsigned __int8 *v5; // rbx
  _QWORD *v6; // r8
  __int64 *v7; // rdi
  unsigned __int64 v8; // rdx
  unsigned __int8 v9; // al
  size_t v10; // rax
  unsigned __int8 *v11; // [rsp+28h] [rbp-138h]
  size_t v12; // [rsp+30h] [rbp-130h]
  unsigned __int8 v13; // [rsp+3Fh] [rbp-121h]
  _QWORD v14[2]; // [rsp+40h] [rbp-120h] BYREF
  _QWORD *v15; // [rsp+50h] [rbp-110h] BYREF
  __int16 v16; // [rsp+60h] [rbp-100h]
  _QWORD v17[2]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v18; // [rsp+80h] [rbp-E0h]
  void *dest; // [rsp+88h] [rbp-D8h]
  int v20; // [rsp+90h] [rbp-D0h]
  unsigned __int64 *v21; // [rsp+98h] [rbp-C8h]
  unsigned __int64 v22[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE v23[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v3 = *(char **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 16LL) + 200LL);
  v11 = &a2[a3];
  if ( &a2[a3] != a2 )
  {
    v5 = a2;
    do
    {
      v9 = *v5;
      v22[0] = (unsigned __int64)v23;
      v13 = v9;
      v22[1] = 0x8000000000LL;
      v20 = 1;
      dest = 0;
      v17[0] = &unk_49EFC48;
      v18 = 0;
      v17[1] = 0;
      v21 = v22;
      sub_16E7A40((__int64)v17, 0, 0, 0);
      v6 = v17;
      if ( v3 )
      {
        v10 = strlen(v3);
        v6 = v17;
        if ( v10 <= v18 - (__int64)dest )
        {
          if ( v10 )
          {
            v12 = v10;
            memcpy(dest, v3, v10);
            dest = (char *)dest + v12;
            v6 = v17;
          }
        }
        else
        {
          v6 = (_QWORD *)sub_16E7EE0((__int64)v17, v3, v10);
        }
      }
      sub_16E7A90((__int64)v6, v13);
      v7 = *(__int64 **)(a1 + 8);
      v8 = *v21;
      v14[1] = *((unsigned int *)v21 + 2);
      v16 = 261;
      v14[0] = v8;
      v15 = v14;
      sub_38DD5A0(v7, (__int64)&v15);
      v17[0] = &unk_49EFD28;
      sub_16E7960((__int64)v17);
      if ( (_BYTE *)v22[0] != v23 )
        _libc_free(v22[0]);
      ++v5;
    }
    while ( v11 != v5 );
  }
}
