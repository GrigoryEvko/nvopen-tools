// Function: sub_3257470
// Address: 0x3257470
//
char __fastcall sub_3257470(__int64 a1)
{
  _UNKNOWN **v2; // rax
  char **v3; // r12
  const char *i; // r13
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  int v8; // edx
  int v9; // ecx
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rdi
  __int64 v13; // rax
  const char *v15[2]; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+30h] [rbp-A0h]
  const char *v17; // [rsp+40h] [rbp-90h] BYREF
  const char *v18; // [rsp+48h] [rbp-88h]
  __int64 v19; // [rsp+50h] [rbp-80h]
  _BYTE v20[120]; // [rsp+58h] [rbp-78h] BYREF

  LOBYTE(v2) = sub_31DA690(*(_QWORD *)(a1 + 8));
  if ( !(_BYTE)v2 )
  {
    v3 = off_49D8CA0;
    for ( i = "__cpp_exception"; ; i = *v3 )
    {
      v5 = *(_QWORD *)(a1 + 8);
      v17 = v20;
      v18 = 0;
      v19 = 60;
      v6 = sub_31DA930(v5);
      LOWORD(v16) = 257;
      if ( *i )
      {
        v15[0] = i;
        LOBYTE(v16) = 3;
      }
      sub_E405D0((__int64)&v17, (char *)v15, v6);
      v7 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 216LL);
      LOWORD(v16) = 261;
      v15[0] = v17;
      v15[1] = v18;
      if ( sub_E65280(v7, v15) )
      {
        v12 = *(_QWORD *)(a1 + 8);
        LOWORD(v16) = 257;
        if ( *i )
        {
          v15[0] = i;
          LOBYTE(v16) = 3;
        }
        v13 = sub_31DE8D0(v12, (unsigned int)v15, v8, v9, v10, v11, (char)v15[0]);
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
          v13,
          0);
      }
      if ( v17 != v20 )
        _libc_free((unsigned __int64)v17);
      ++v3;
      v2 = &off_49D8CB0;
      if ( &off_49D8CB0 == (_UNKNOWN **)v3 )
        break;
    }
  }
  return (char)v2;
}
