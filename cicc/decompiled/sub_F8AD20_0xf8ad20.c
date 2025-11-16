// Function: sub_F8AD20
// Address: 0xf8ad20
//
_QWORD *__fastcall sub_F8AD20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // r15
  unsigned int v10; // eax
  __int64 v11; // rdi
  _QWORD *v12; // r14
  _QWORD **v14; // rdx
  int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int16 v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+18h] [rbp-98h]
  const char *v25; // [rsp+20h] [rbp-90h] BYREF
  char v26; // [rsp+40h] [rbp-70h]
  char v27; // [rsp+41h] [rbp-6Fh]
  _BYTE v28[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v29; // [rsp+70h] [rbp-40h]

  v4 = a1 + 520;
  v5 = *(_QWORD *)(a2 + 40);
  sub_D5F1F0(a1 + 520, a3);
  v6 = sub_F894B0(a1, v5);
  v7 = *(_QWORD *)(a2 + 48);
  v8 = v6;
  sub_D5F1F0(a1 + 520, a3);
  v9 = sub_F894B0(a1, v7);
  sub_D5F1F0(a1 + 520, a3);
  v10 = sub_B52870(*(_DWORD *)(a2 + 36));
  v27 = 1;
  v11 = *(_QWORD *)(a1 + 600);
  v23 = v10;
  v25 = "ident.check";
  v26 = 3;
  v12 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v11 + 56LL))(
                    v11,
                    v10,
                    v8,
                    v9);
  if ( !v12 )
  {
    v29 = 257;
    v12 = sub_BD2C40(72, unk_3F10FD0);
    if ( v12 )
    {
      v14 = *(_QWORD ***)(v8 + 8);
      v15 = *((unsigned __int8 *)v14 + 8);
      if ( (unsigned int)(v15 - 17) > 1 )
      {
        v17 = sub_BCB2A0(*v14);
      }
      else
      {
        BYTE4(v24) = (_BYTE)v15 == 18;
        LODWORD(v24) = *((_DWORD *)v14 + 8);
        v16 = (__int64 *)sub_BCB2A0(*v14);
        v17 = sub_BCE1B0(v16, v24);
      }
      sub_B523C0((__int64)v12, v17, 53, v23, v8, v9, (__int64)v28, 0, 0, 0);
    }
    (*(void (__fastcall **)(_QWORD, _QWORD *, const char **, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
      *(_QWORD *)(a1 + 608),
      v12,
      &v25,
      *(_QWORD *)(v4 + 56),
      *(_QWORD *)(v4 + 64));
    v18 = *(_QWORD *)(a1 + 520);
    v19 = v18 + 16LL * *(unsigned int *)(a1 + 528);
    while ( v19 != v18 )
    {
      v20 = *(_QWORD *)(v18 + 8);
      v21 = *(_DWORD *)v18;
      v18 += 16;
      sub_B99FD0((__int64)v12, v21, v20);
    }
  }
  return v12;
}
