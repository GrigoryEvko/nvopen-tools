// Function: sub_CCDDA0
// Address: 0xccdda0
//
__int64 __fastcall sub_CCDDA0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  char *v3; // rdx
  __int64 v4; // rax
  void (__fastcall *v5)(__int64, char **, _QWORD); // r13
  unsigned int v6; // eax
  __int64 result; // rax
  __int64 v8; // rax
  void (__fastcall *v9)(__int64, void **, _QWORD); // rbx
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 (__fastcall *v13)(__int64, char **); // rcx
  char *v14[2]; // [rsp+0h] [rbp-110h] BYREF
  void *v15; // [rsp+10h] [rbp-100h] BYREF
  __int64 v16; // [rsp+18h] [rbp-F8h]
  __int64 v17; // [rsp+20h] [rbp-F0h]
  __int64 v18; // [rsp+28h] [rbp-E8h]
  __int64 v19; // [rsp+30h] [rbp-E0h]
  __int64 v20; // [rsp+38h] [rbp-D8h]
  char **v21; // [rsp+40h] [rbp-D0h]
  char *v22; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v23; // [rsp+58h] [rbp-B8h]
  __int64 v24; // [rsp+60h] [rbp-B0h]
  char v25; // [rsp+68h] [rbp-A8h] BYREF
  __int16 v26; // [rsp+70h] [rbp-A0h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v20 = 0x100000000LL;
    v21 = &v22;
    v15 = &unk_49DD288;
    v22 = &v25;
    v23 = 0;
    v24 = 128;
    v16 = 2;
    v17 = 0;
    v18 = 0;
    v19 = 0;
    sub_CB5980((__int64)&v15, 0, 0, 0);
    v2 = sub_CB0A70(a1);
    sub_CB2AD0((__int64)a2, v2, (__int64)&v15);
    v3 = v21[1];
    v14[0] = *v21;
    v4 = *(_QWORD *)a1;
    v14[1] = v3;
    v5 = *(void (__fastcall **)(__int64, char **, _QWORD))(v4 + 216);
    v6 = sub_C2FE50(v14[0], (__int64)v3, 1);
    v5(a1, v14, v6);
    v15 = &unk_49DD388;
    result = (__int64)sub_CB5840((__int64)&v15);
    if ( v22 != &v25 )
      return _libc_free(v22, v14);
  }
  else
  {
    v8 = *(_QWORD *)a1;
    v15 = 0;
    v16 = 0;
    v9 = *(void (__fastcall **)(__int64, void **, _QWORD))(v8 + 216);
    v10 = sub_C2FE50(0, 0, 1);
    v9(a1, &v15, v10);
    v11 = sub_CB0A70(a1);
    result = sub_CB2B30((__int64)v15, v16, v11, a2);
    if ( v12 )
    {
      v13 = *(__int64 (__fastcall **)(__int64, char **))(*(_QWORD *)a1 + 248LL);
      v26 = 261;
      v22 = (char *)result;
      v23 = v12;
      return v13(a1, &v22);
    }
  }
  return result;
}
