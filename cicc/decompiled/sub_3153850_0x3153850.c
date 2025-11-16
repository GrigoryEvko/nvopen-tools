// Function: sub_3153850
// Address: 0x3153850
//
void __fastcall sub_3153850(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  const char *v4; // rax
  __int64 v5; // rdx
  void (__fastcall *v6)(__int64, const char **); // rcx
  __int64 v7; // rax
  const char *v8; // rdx
  __int64 v9; // rax
  _QWORD v10[2]; // [rsp+0h] [rbp-110h] BYREF
  void *v11; // [rsp+10h] [rbp-100h] BYREF
  __int64 v12; // [rsp+18h] [rbp-F8h]
  __int64 v13; // [rsp+20h] [rbp-F0h]
  __int64 v14; // [rsp+28h] [rbp-E8h]
  __int64 v15; // [rsp+30h] [rbp-E0h]
  __int64 v16; // [rsp+38h] [rbp-D8h]
  const char **v17; // [rsp+40h] [rbp-D0h]
  const char *v18; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+58h] [rbp-B8h]
  __int64 v20; // [rsp+60h] [rbp-B0h]
  char v21; // [rsp+68h] [rbp-A8h] BYREF
  __int16 v22; // [rsp+70h] [rbp-A0h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v16 = 0x100000000LL;
    v17 = &v18;
    v11 = &unk_49DD288;
    v18 = &v21;
    v19 = 0;
    v20 = 128;
    v12 = 2;
    v13 = 0;
    v14 = 0;
    v15 = 0;
    sub_CB5980((__int64)&v11, 0, 0, 0);
    v7 = sub_CB0A70(a1);
    sub_CB2DC0(a2, v7, (__int64)&v11);
    v8 = v17[1];
    v10[0] = *v17;
    v9 = *(_QWORD *)a1;
    v10[1] = v8;
    (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(v9 + 216))(a1, v10, 0);
    v11 = &unk_49DD388;
    sub_CB5840((__int64)&v11);
    if ( v18 != &v21 )
      _libc_free((unsigned __int64)v18);
  }
  else
  {
    v2 = *(_QWORD *)a1;
    v11 = 0;
    v12 = 0;
    (*(void (__fastcall **)(__int64, void **, _QWORD))(v2 + 216))(a1, &v11, 0);
    v3 = sub_CB0A70(a1);
    v4 = sub_CB2DD0((__int64)v11, v12, v3, a2);
    if ( v5 )
    {
      v6 = *(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 248LL);
      v22 = 261;
      v18 = v4;
      v19 = v5;
      v6(a1, &v18);
    }
  }
}
