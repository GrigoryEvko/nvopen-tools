// Function: sub_29CB310
// Address: 0x29cb310
//
void __fastcall sub_29CB310(__int64 *a1, unsigned __int8 *a2, size_t a3, _QWORD **a4)
{
  _QWORD *v6; // r12
  __int64 v7; // rcx
  __int64 v8; // r14
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // r10
  __int64 v12; // rax
  __int64 v13; // rcx
  char v14; // al
  __int64 v15; // rax
  _QWORD *v16; // r14
  __int64 v17; // r10
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v22; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v23; // [rsp+18h] [rbp-78h]
  int v24; // [rsp+20h] [rbp-70h]
  int v25; // [rsp+24h] [rbp-6Ch]
  int v26; // [rsp+28h] [rbp-68h]
  char v27; // [rsp+2Ch] [rbp-64h]
  _QWORD v28[3]; // [rsp+30h] [rbp-60h] BYREF
  char *v29; // [rsp+48h] [rbp-48h]
  __int64 v30; // [rsp+50h] [rbp-40h]
  int v31; // [rsp+58h] [rbp-38h]
  char v32; // [rsp+5Ch] [rbp-34h]
  char v33; // [rsp+60h] [rbp-30h] BYREF

  v6 = *a4;
  *a4 = 0;
  if ( !(unsigned __int8)sub_29C0DC0((char *)a2, a3) )
  {
    v24 = 2;
    v23 = v28;
    v29 = &v33;
    v26 = 0;
    v27 = 1;
    v28[2] = 0;
    v30 = 2;
    v31 = 0;
    v32 = 1;
    v25 = 1;
    v28[0] = &unk_4F82408;
    v22 = 1;
    if ( !v6 )
      return;
    if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD *))(*v6 + 24LL))(v6) == &unk_4C5D161 )
    {
      v7 = *a1;
      v8 = v6[1];
      v9 = *(_QWORD **)(v8 + 40);
      v10 = v8 + 56;
      v11 = *(_QWORD *)(v8 + 64);
      if ( *(_DWORD *)(*a1 + 24) == 1 )
        sub_29C57C0(v9, v10, v11, (__int64)a2, a3, 1u, "CheckFunctionDebugify", 0x15u, *(_DWORD **)(v7 + 32));
      else
        sub_29C8000(
          (__int64)v9,
          v10,
          v11,
          *(_QWORD *)(v7 + 16),
          "CheckModuleDebugify (original debuginfo)",
          0x28u,
          a2,
          a3,
          *(unsigned __int8 **)v7,
          *(_QWORD *)(v7 + 8));
      v12 = sub_BC0510(a1[1], &unk_4F82418, *(_QWORD *)(v8 + 40));
      sub_BBE020(*(_QWORD *)(v12 + 8), v8, (__int64)&v22, v13);
      v14 = v32;
    }
    else
    {
      if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD *))(*v6 + 24LL))(v6) == &unk_4C5D162 )
      {
        v15 = *a1;
        v16 = (_QWORD *)v6[1];
        v17 = v16[4];
        v18 = (__int64)(v16 + 3);
        if ( *(_DWORD *)(*a1 + 24) == 1 )
          sub_29C57C0(v16, v17, v18, (__int64)a2, a3, 1u, "CheckModuleDebugify", 0x13u, *(_DWORD **)(v15 + 32));
        else
          sub_29C8000(
            (__int64)v16,
            v17,
            v18,
            *(_QWORD *)(v15 + 16),
            "CheckModuleDebugify (original debuginfo)",
            0x28u,
            a2,
            a3,
            *(unsigned __int8 **)v15,
            *(_QWORD *)(v15 + 8));
        sub_BBD520(a1[1], (__int64)v16, (__int64)&v22, v19);
      }
      v14 = v32;
    }
    if ( !v14 )
      _libc_free((unsigned __int64)v29);
    if ( !v27 )
    {
      _libc_free((unsigned __int64)v23);
      goto LABEL_3;
    }
  }
  if ( v6 )
LABEL_3:
    (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
}
