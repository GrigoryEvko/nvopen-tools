// Function: sub_15D6470
// Address: 0x15d6470
//
void __fastcall sub_15D6470(__int64 a1, __int64 a2, int a3, unsigned int a4, __int64 a5, int a6)
{
  __int64 v6; // r14
  unsigned int v9; // eax
  char *v10; // rdx
  __int64 v11; // rax
  char **v12; // r12
  char **v13; // r14
  char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r8
  unsigned int v17; // ecx
  __int64 v18; // rdi
  char *v19; // r10
  int v20; // edi
  int v21; // edx
  __int64 v23; // [rsp+38h] [rbp-2C8h]
  __int64 v24; // [rsp+40h] [rbp-2C0h]
  char **v27; // [rsp+50h] [rbp-2B0h]
  __int64 v28; // [rsp+58h] [rbp-2A8h] BYREF
  char *v29; // [rsp+60h] [rbp-2A0h] BYREF
  __int64 v30; // [rsp+68h] [rbp-298h] BYREF
  char **v31; // [rsp+70h] [rbp-290h] BYREF
  int v32; // [rsp+78h] [rbp-288h]
  char v33; // [rsp+80h] [rbp-280h] BYREF
  _QWORD *v34; // [rsp+C0h] [rbp-240h] BYREF
  __int64 v35; // [rsp+C8h] [rbp-238h]
  _QWORD v36[70]; // [rsp+D0h] [rbp-230h] BYREF

  v6 = a1 + 24;
  v28 = a2;
  v34 = v36;
  v36[0] = a2;
  v30 = a2;
  v35 = 0x4000000001LL;
  if ( (unsigned __int8)sub_15CE6E0(a1 + 24, &v30, &v31) )
    *(_DWORD *)(sub_15D4720(v6, &v28) + 12) = a6;
  v9 = v35;
  if ( (_DWORD)v35 )
  {
LABEL_6:
    while ( 1 )
    {
      v10 = (char *)v34[v9 - 1];
      LODWORD(v35) = v9 - 1;
      v29 = v10;
      v11 = sub_15D4720(v6, (__int64 *)&v29);
      if ( !*(_DWORD *)(v11 + 8) )
        break;
LABEL_5:
      v9 = v35;
      if ( !(_DWORD)v35 )
        goto LABEL_21;
    }
    *(_DWORD *)(v11 + 16) = ++a3;
    *(_DWORD *)(v11 + 8) = a3;
    *(_QWORD *)(v11 + 24) = v29;
    sub_15CE600(a1, &v29);
    sub_15CF0D0((__int64)&v31, (__int64)v29, *(_QWORD *)(a1 + 56));
    v12 = &v31[v32];
    if ( v31 == v12 )
      goto LABEL_19;
    v23 = v6;
    v13 = v31;
    v27 = &v31[v32];
    while ( 1 )
    {
      v14 = *v13;
      v15 = *(unsigned int *)(a1 + 48);
      v30 = (__int64)*v13;
      if ( !(_DWORD)v15 )
        goto LABEL_9;
      v16 = *(_QWORD *)(a1 + 32);
      v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = v16 + 72LL * v17;
      v19 = *(char **)v18;
      if ( v14 != *(char **)v18 )
        break;
LABEL_14:
      if ( v18 == v16 + 72 * v15 || !*(_DWORD *)(v18 + 8) )
        goto LABEL_9;
      if ( v14 == v29 )
      {
LABEL_11:
        if ( v27 == ++v13 )
          goto LABEL_18;
      }
      else
      {
        ++v13;
        sub_15CDD90(v18 + 40, &v29);
        if ( v27 == v13 )
        {
LABEL_18:
          v6 = v23;
          v12 = v31;
LABEL_19:
          if ( v12 == (char **)&v33 )
            goto LABEL_5;
          _libc_free((unsigned __int64)v12);
          v9 = v35;
          if ( !(_DWORD)v35 )
            goto LABEL_21;
          goto LABEL_6;
        }
      }
    }
    v20 = 1;
    while ( v19 != (char *)-8LL )
    {
      v21 = v20 + 1;
      v17 = (v15 - 1) & (v20 + v17);
      v18 = v16 + 72LL * v17;
      v19 = *(char **)v18;
      if ( v14 == *(char **)v18 )
        goto LABEL_14;
      v20 = v21;
    }
LABEL_9:
    if ( *(_DWORD *)(sub_15CC960(a5, (__int64)v14) + 16) > a4 )
    {
      v24 = sub_15D4720(v23, &v30);
      sub_15CDD90((__int64)&v34, &v30);
      *(_DWORD *)(v24 + 12) = a3;
      sub_15CDD90(v24 + 40, &v29);
    }
    goto LABEL_11;
  }
LABEL_21:
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
