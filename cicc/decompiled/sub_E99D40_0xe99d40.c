// Function: sub_E99D40
// Address: 0xe99d40
//
__int64 __fastcall sub_E99D40(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  char *v3; // r14
  __int64 result; // rax
  unsigned __int8 *v6; // rbx
  _QWORD *v7; // r8
  __int64 *v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 v11; // al
  size_t v12; // rax
  unsigned __int8 *v13; // [rsp+18h] [rbp-158h]
  size_t v14; // [rsp+20h] [rbp-150h]
  unsigned __int8 v15; // [rsp+2Fh] [rbp-141h]
  _QWORD v16[4]; // [rsp+30h] [rbp-140h] BYREF
  __int16 v17; // [rsp+50h] [rbp-120h]
  _QWORD v18[3]; // [rsp+60h] [rbp-110h] BYREF
  __int64 v19; // [rsp+78h] [rbp-F8h]
  void *dest; // [rsp+80h] [rbp-F0h]
  __int64 v21; // [rsp+88h] [rbp-E8h]
  __int64 *v22; // [rsp+90h] [rbp-E0h]
  _QWORD v23[3]; // [rsp+A0h] [rbp-D0h] BYREF
  _BYTE v24[184]; // [rsp+B8h] [rbp-B8h] BYREF

  v3 = *(char **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 152LL) + 224LL);
  result = (__int64)&a2[a3];
  v13 = &a2[a3];
  if ( &a2[a3] != a2 )
  {
    v6 = a2;
    do
    {
      v11 = *v6;
      v23[0] = v24;
      v15 = v11;
      v21 = 0x100000000LL;
      v23[1] = 0;
      v23[2] = 128;
      v18[0] = &unk_49DD288;
      v18[1] = 2;
      v18[2] = 0;
      v19 = 0;
      dest = 0;
      v22 = v23;
      sub_CB5980((__int64)v18, 0, 0, 0);
      v7 = v18;
      if ( v3 )
      {
        v12 = strlen(v3);
        v7 = v18;
        if ( v12 <= v19 - (__int64)dest )
        {
          if ( v12 )
          {
            v14 = v12;
            memcpy(dest, v3, v12);
            dest = (char *)dest + v14;
            v7 = v18;
          }
        }
        else
        {
          v7 = (_QWORD *)sub_CB6200((__int64)v18, (unsigned __int8 *)v3, v12);
        }
      }
      sub_CB59D0((__int64)v7, v15);
      v8 = *(__int64 **)(a1 + 8);
      v9 = v22[1];
      v10 = *v22;
      v17 = 261;
      v16[0] = v10;
      v16[1] = v9;
      sub_E99A90(v8, (__int64)v16);
      v18[0] = &unk_49DD388;
      result = (__int64)sub_CB5840((__int64)v18);
      if ( (_BYTE *)v23[0] != v24 )
        result = _libc_free(v23[0], v16);
      ++v6;
    }
    while ( v13 != v6 );
  }
  return result;
}
