// Function: sub_25B59B0
// Address: 0x25b59b0
//
__int64 __fastcall sub_25B59B0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  _BYTE *v6; // r15
  unsigned __int64 v7; // r12
  _QWORD *v8; // rax
  bool v9; // zf
  int v10; // r8d
  int v11; // r9d
  unsigned __int8 v13; // al
  _QWORD *v14; // rdi
  _QWORD *v16; // [rsp+10h] [rbp-140h] BYREF
  __int64 v17; // [rsp+18h] [rbp-138h]
  _QWORD *v18; // [rsp+20h] [rbp-130h] BYREF
  size_t v19; // [rsp+28h] [rbp-128h]
  _BYTE v20[16]; // [rsp+30h] [rbp-120h] BYREF
  unsigned __int64 v21[2]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v22[6]; // [rsp+50h] [rbp-100h] BYREF
  _QWORD v23[8]; // [rsp+80h] [rbp-D0h] BYREF
  void *src; // [rsp+C0h] [rbp-90h] BYREF
  size_t n; // [rsp+C8h] [rbp-88h]
  const char *v26; // [rsp+D0h] [rbp-80h]
  __int64 v27; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v28; // [rsp+F8h] [rbp-58h]
  char v29; // [rsp+10Ch] [rbp-44h]

  if ( sub_BA8CD0(a3, (__int64)"llvm.embedded.module", 0x14u, 1) )
    sub_C64ED0("Can only embed the module once", 0);
  v6 = *(_BYTE **)(a3 + 232);
  v7 = *(_QWORD *)(a3 + 240);
  v21[0] = (unsigned __int64)v22;
  if ( &v6[v7] && !v6 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  src = (void *)v7;
  if ( v7 > 0xF )
  {
    v21[0] = sub_22409D0((__int64)v21, (unsigned __int64 *)&src, 0);
    v14 = (_QWORD *)v21[0];
    v22[0] = src;
  }
  else
  {
    if ( v7 == 1 )
    {
      LOBYTE(v22[0]) = *v6;
      v8 = v22;
      goto LABEL_7;
    }
    if ( !v7 )
    {
      v8 = v22;
      goto LABEL_7;
    }
    v14 = v22;
  }
  memcpy(v14, v6, v7);
  v7 = (unsigned __int64)src;
  v8 = (_QWORD *)v21[0];
LABEL_7:
  v21[1] = v7;
  *((_BYTE *)v8 + v7) = 0;
  v9 = *(_DWORD *)(a3 + 284) == 3;
  v22[2] = *(_QWORD *)(a3 + 264);
  v22[3] = *(_QWORD *)(a3 + 272);
  v22[4] = *(_QWORD *)(a3 + 280);
  if ( !v9 )
    sub_C64ED0("EmbedBitcode pass currently only supports ELF object format", 0);
  v23[5] = 0x100000000LL;
  v18 = v20;
  v19 = 0;
  v23[0] = &unk_49DD210;
  v20[0] = 0;
  memset(&v23[1], 0, 32);
  v23[6] = &v18;
  sub_CB5980((__int64)v23, 0, 0, 0);
  if ( *a2 )
  {
    v16 = v23;
    v17 = 0;
    sub_26F5D00(&src, &v16);
    if ( v29 )
      goto LABEL_10;
LABEL_19:
    _libc_free(v28);
    if ( BYTE4(v27) )
      goto LABEL_11;
    goto LABEL_20;
  }
  v13 = a2[1];
  v16 = v23;
  LOBYTE(v17) = 0;
  *(_WORD *)((char *)&v17 + 1) = v13;
  sub_A3CA60((__int64)&src, (__int64)&v16, a3, a4);
  if ( !v29 )
    goto LABEL_19;
LABEL_10:
  if ( BYTE4(v27) )
    goto LABEL_11;
LABEL_20:
  _libc_free(n);
LABEL_11:
  v27 = 10;
  src = v18;
  n = v19;
  v26 = "ModuleData";
  sub_2A41DE0(a3, (int)".llvm.lto", 9, 0, v10, v11, v18, v19);
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  v23[0] = &unk_49DD210;
  sub_CB5840((__int64)v23);
  if ( v18 != (_QWORD *)v20 )
    j_j___libc_free_0((unsigned __int64)v18);
  if ( (_QWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0]);
  return a1;
}
