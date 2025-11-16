// Function: sub_15D5190
// Address: 0x15d5190
//
__int64 __fastcall sub_15D5190(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int8 (__fastcall *a4)(char *),
        int a5)
{
  __int64 v5; // r14
  unsigned int v8; // eax
  char *v9; // rdx
  __int64 v10; // rax
  char **v11; // r12
  char **v12; // r14
  char *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 v17; // rdi
  char *v18; // r10
  int v20; // edi
  int v21; // edx
  __int64 v22; // [rsp+28h] [rbp-2C8h]
  __int64 v23; // [rsp+30h] [rbp-2C0h]
  __int64 v26; // [rsp+48h] [rbp-2A8h] BYREF
  char *v27; // [rsp+50h] [rbp-2A0h] BYREF
  __int64 v28; // [rsp+58h] [rbp-298h] BYREF
  char **v29; // [rsp+60h] [rbp-290h] BYREF
  int v30; // [rsp+68h] [rbp-288h]
  char v31; // [rsp+70h] [rbp-280h] BYREF
  _QWORD *v32; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v33; // [rsp+B8h] [rbp-238h]
  _QWORD v34[70]; // [rsp+C0h] [rbp-230h] BYREF

  v5 = a1 + 24;
  v32 = v34;
  v26 = a2;
  v34[0] = a2;
  v33 = 0x4000000001LL;
  v28 = a2;
  if ( (unsigned __int8)sub_15CE6E0(a1 + 24, &v28, &v29) )
    *(_DWORD *)(sub_15D4720(v5, &v26) + 12) = a5;
  v8 = v33;
  if ( (_DWORD)v33 )
  {
LABEL_6:
    while ( 1 )
    {
      v9 = (char *)v32[v8 - 1];
      LODWORD(v33) = v8 - 1;
      v27 = v9;
      v10 = sub_15D4720(v5, (__int64 *)&v27);
      if ( !*(_DWORD *)(v10 + 8) )
        break;
LABEL_5:
      v8 = v33;
      if ( !(_DWORD)v33 )
        goto LABEL_21;
    }
    *(_DWORD *)(v10 + 16) = ++a3;
    *(_DWORD *)(v10 + 8) = a3;
    *(_QWORD *)(v10 + 24) = v27;
    sub_15CE600(a1, &v27);
    sub_15CF0D0((__int64)&v29, (__int64)v27, *(_QWORD *)(a1 + 56));
    v11 = &v29[v30];
    if ( v29 == v11 )
      goto LABEL_19;
    v22 = v5;
    v12 = v29;
    while ( 1 )
    {
      v13 = *v12;
      v14 = *(unsigned int *)(a1 + 48);
      v28 = (__int64)*v12;
      if ( !(_DWORD)v14 )
        goto LABEL_9;
      v15 = *(_QWORD *)(a1 + 32);
      v16 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v17 = v15 + 72LL * v16;
      v18 = *(char **)v17;
      if ( v13 != *(char **)v17 )
        break;
LABEL_14:
      if ( v17 == v15 + 72 * v14 || !*(_DWORD *)(v17 + 8) )
        goto LABEL_9;
      if ( v13 == v27 )
      {
LABEL_11:
        if ( v11 == ++v12 )
          goto LABEL_18;
      }
      else
      {
        ++v12;
        sub_15CDD90(v17 + 40, &v27);
        if ( v11 == v12 )
        {
LABEL_18:
          v5 = v22;
          v11 = v29;
LABEL_19:
          if ( v11 == (char **)&v31 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v11);
          v8 = v33;
          if ( !(_DWORD)v33 )
            goto LABEL_21;
          goto LABEL_6;
        }
      }
    }
    v20 = 1;
    while ( v18 != (char *)-8LL )
    {
      v21 = v20 + 1;
      v16 = (v14 - 1) & (v20 + v16);
      v17 = v15 + 72LL * v16;
      v18 = *(char **)v17;
      if ( v13 == *(char **)v17 )
        goto LABEL_14;
      v20 = v21;
    }
LABEL_9:
    if ( a4(v27) )
    {
      v23 = sub_15D4720(v22, &v28);
      sub_15CDD90((__int64)&v32, &v28);
      *(_DWORD *)(v23 + 12) = a3;
      sub_15CDD90(v23 + 40, &v27);
    }
    goto LABEL_11;
  }
LABEL_21:
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  return a3;
}
