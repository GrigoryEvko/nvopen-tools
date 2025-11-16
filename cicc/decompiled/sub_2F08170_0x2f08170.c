// Function: sub_2F08170
// Address: 0x2f08170
//
void __fastcall sub_2F08170(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  const char *v10; // rax
  __int64 v11; // rdx
  void (__fastcall *v12)(__int64, const char **); // rcx
  __int64 v13; // rax
  const char *v14; // rdx
  __int64 v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-120h] BYREF
  void *v17; // [rsp+10h] [rbp-110h] BYREF
  __int64 v18; // [rsp+18h] [rbp-108h]
  __int64 v19; // [rsp+20h] [rbp-100h]
  __int64 v20; // [rsp+28h] [rbp-F8h]
  __int64 v21; // [rsp+30h] [rbp-F0h]
  __int64 v22; // [rsp+38h] [rbp-E8h]
  const char **v23; // [rsp+40h] [rbp-E0h]
  const char *v24; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v25; // [rsp+58h] [rbp-C8h]
  __int64 v26; // [rsp+60h] [rbp-C0h]
  char v27; // [rsp+68h] [rbp-B8h] BYREF
  __int16 v28; // [rsp+70h] [rbp-B0h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v22 = 0x100000000LL;
    v23 = &v24;
    v17 = &unk_49DD288;
    v24 = &v27;
    v25 = 0;
    v26 = 128;
    v18 = 2;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    sub_CB5980((__int64)&v17, 0, 0, 0);
    v13 = sub_CB0A70(a1);
    sub_CB2D50((unsigned int *)a2, v13, (__int64)&v17);
    v14 = v23[1];
    v16[0] = *v23;
    v15 = *(_QWORD *)a1;
    v16[1] = v14;
    (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(v15 + 216))(a1, v16, 0);
    v17 = &unk_49DD388;
    sub_CB5840((__int64)&v17);
    if ( v24 != &v27 )
      _libc_free((unsigned __int64)v24);
  }
  else
  {
    v2 = *(_QWORD *)a1;
    v17 = 0;
    v18 = 0;
    (*(void (__fastcall **)(__int64, void **, _QWORD))(v2 + 216))(a1, &v17, 0);
    v3 = sub_CB0A70(a1);
    v4 = (__int64)v17;
    v5 = v18;
    v6 = v3;
    v7 = sub_CB1000(v3);
    if ( v7 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      v9 = *(_QWORD *)(v7 + 24);
      *(_QWORD *)(a2 + 8) = v8;
      *(_QWORD *)(a2 + 16) = v9;
    }
    v10 = sub_CB2D60(v4, v5, v6, (_DWORD *)a2);
    if ( v11 )
    {
      v12 = *(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 248LL);
      v28 = 261;
      v24 = v10;
      v25 = v11;
      v12(a1, &v24);
    }
  }
}
