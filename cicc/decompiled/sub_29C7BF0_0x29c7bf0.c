// Function: sub_29C7BF0
// Address: 0x29c7bf0
//
void __fastcall sub_29C7BF0(__int64 *a1, char *a2, size_t a3, _QWORD **a4)
{
  _QWORD *v5; // r12
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rcx
  char v9; // al
  __int64 v10; // r10
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r10
  __int64 *v14; // rdi
  __int64 v15; // [rsp+0h] [rbp-C0h]
  _BYTE v16[16]; // [rsp+10h] [rbp-B0h] BYREF
  void (__fastcall *v17)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-A0h]
  __int64 v18; // [rsp+30h] [rbp-90h] BYREF
  _QWORD *v19; // [rsp+38h] [rbp-88h]
  int v20; // [rsp+40h] [rbp-80h]
  int v21; // [rsp+44h] [rbp-7Ch]
  int v22; // [rsp+48h] [rbp-78h]
  char v23; // [rsp+4Ch] [rbp-74h]
  _QWORD v24[3]; // [rsp+50h] [rbp-70h] BYREF
  char *v25; // [rsp+68h] [rbp-58h]
  __int64 v26; // [rsp+70h] [rbp-50h]
  int v27; // [rsp+78h] [rbp-48h]
  char v28; // [rsp+7Ch] [rbp-44h]
  char v29; // [rsp+80h] [rbp-40h] BYREF

  v5 = *a4;
  *a4 = 0;
  if ( !(unsigned __int8)sub_29C0DC0(a2, a3) )
  {
    v20 = 2;
    v19 = v24;
    v25 = &v29;
    v22 = 0;
    v23 = 1;
    v24[2] = 0;
    v26 = 2;
    v27 = 0;
    v28 = 1;
    v21 = 1;
    v24[0] = &unk_4F82408;
    v18 = 1;
    if ( !v5 )
      return;
    if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD *))(*v5 + 24LL))(v5) == &unk_4C5D161 )
    {
      v6 = v5[1];
      sub_29C7B10(v6, *(_DWORD *)(*a1 + 24), *(_QWORD *)(*a1 + 16));
      v7 = sub_BC0510(a1[1], &unk_4F82418, *(_QWORD *)(v6 + 40));
      sub_BBE020(*(_QWORD *)(v7 + 8), v6, (__int64)&v18, v8);
      v9 = v28;
    }
    else
    {
      if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD *))(*v5 + 24LL))(v5) == &unk_4C5D162 )
      {
        v10 = v5[1];
        v11 = *(_QWORD *)(v10 + 32);
        if ( *(_DWORD *)(*a1 + 24) == 1 )
        {
          v14 = (__int64 *)v5[1];
          v17 = 0;
          sub_29C2F90(v14, v11, v10 + 24, "ModuleDebugify: ", 0x10u, (__int64)v16);
          v13 = (__int64)v14;
          if ( v17 )
          {
            v17(v16, v16, 3);
            v13 = (__int64)v14;
          }
        }
        else
        {
          v15 = v5[1];
          sub_29C70F0(v10, v11, v10 + 24, *(_QWORD *)(*a1 + 16), "ModuleDebugify (original debuginfo)", 0x23u);
          v13 = v15;
        }
        sub_BBD520(a1[1], v13, (__int64)&v18, v12);
      }
      v9 = v28;
    }
    if ( !v9 )
      _libc_free((unsigned __int64)v25);
    if ( !v23 )
    {
      _libc_free((unsigned __int64)v19);
      goto LABEL_3;
    }
  }
  if ( v5 )
LABEL_3:
    (*(void (__fastcall **)(_QWORD *))(*v5 + 8LL))(v5);
}
