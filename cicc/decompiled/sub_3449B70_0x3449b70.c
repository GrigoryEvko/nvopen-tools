// Function: sub_3449B70
// Address: 0x3449b70
//
bool __fastcall sub_3449B70(_DWORD *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned int v11; // edx
  __int64 v12; // r12
  unsigned __int16 *v13; // rdx
  int v14; // eax
  __int64 v15; // r14
  __int64 v16; // rdx
  unsigned int v17; // edx
  bool result; // al
  unsigned int v19; // ebx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // r12
  __int16 v26; // ax
  __int64 v27; // rdx
  unsigned __int16 v28; // cx
  int v29; // eax
  unsigned int v30; // ebx
  __int64 v31; // rdx
  __int64 v32; // rax
  char v33; // bl
  char *v34; // rax
  bool v35; // [rsp-79h] [rbp-79h]
  char *v36; // [rsp-78h] [rbp-78h] BYREF
  unsigned int v37; // [rsp-70h] [rbp-70h]
  __int16 v38; // [rsp-68h] [rbp-68h] BYREF
  __int64 v39; // [rsp-60h] [rbp-60h]
  __int64 v40; // [rsp-58h] [rbp-58h]
  __int64 v41; // [rsp-50h] [rbp-50h]
  char *v42; // [rsp-48h] [rbp-48h] BYREF
  __int64 v43; // [rsp-40h] [rbp-40h]

  if ( !a2 )
    return 0;
  v37 = 1;
  v36 = 0;
  v8 = sub_33DFBC0(a2, a3, 0, 1u, a5, a6);
  if ( !v8 )
  {
    v17 = v37;
    result = 0;
    goto LABEL_17;
  }
  v9 = *(_QWORD *)(v8 + 96);
  v10 = v9 + 24;
  v11 = *(_DWORD *)(v9 + 32);
  if ( v11 <= 0x40 )
  {
    v34 = *(char **)(v9 + 24);
    v37 = v11;
    v36 = v34;
  }
  else
  {
    sub_C43990((__int64)&v36, v10);
  }
  v12 = 16LL * a3;
  v13 = (unsigned __int16 *)(v12 + *(_QWORD *)(a2 + 48));
  v14 = *v13;
  v15 = *((_QWORD *)v13 + 1);
  v38 = v14;
  v39 = v15;
  if ( !(_WORD)v14 )
  {
    if ( !sub_30070B0((__int64)&v38) )
    {
      v43 = v15;
      LOWORD(v42) = 0;
      goto LABEL_26;
    }
    LOWORD(v14) = sub_3009970((__int64)&v38, v10, v20, v21, v22);
LABEL_38:
    LOWORD(v42) = v14;
    v43 = v31;
    if ( (_WORD)v14 )
      goto LABEL_9;
LABEL_26:
    v23 = sub_3007260((__int64)&v42);
    v24 = v16;
    v40 = v23;
    LODWORD(v16) = v23;
    v41 = v24;
    if ( (unsigned int)v23 >= v37 )
      goto LABEL_27;
    goto LABEL_12;
  }
  if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
  {
    LOWORD(v14) = word_4456580[v14 - 1];
    v31 = 0;
    goto LABEL_38;
  }
  LOWORD(v42) = v14;
  v43 = v15;
LABEL_9:
  if ( (_WORD)v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    goto LABEL_53;
  v16 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v14 - 16];
  if ( (unsigned int)v16 < v37 )
  {
LABEL_12:
    sub_C44740((__int64)&v42, &v36, v16);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0((unsigned __int64)v36);
    v36 = v42;
    v37 = v43;
  }
LABEL_27:
  v25 = *(_QWORD *)(a2 + 48) + v12;
  v26 = *(_WORD *)v25;
  v27 = *(_QWORD *)(v25 + 8);
  LOWORD(v42) = v26;
  v43 = v27;
  if ( !v26 )
  {
    v33 = sub_3007030((__int64)&v42);
    if ( sub_30070B0((__int64)&v42) )
      goto LABEL_49;
    if ( !v33 )
      goto LABEL_31;
LABEL_47:
    v29 = a1[16];
    goto LABEL_32;
  }
  v28 = v26 - 17;
  if ( (unsigned __int16)(v26 - 10) > 6u && (unsigned __int16)(v26 - 126) > 0x31u )
  {
    if ( v28 > 0xD3u )
    {
LABEL_31:
      v29 = a1[15];
      goto LABEL_32;
    }
    goto LABEL_49;
  }
  if ( v28 > 0xD3u )
    goto LABEL_47;
LABEL_49:
  v29 = a1[17];
LABEL_32:
  if ( v29 == 1 )
  {
    v19 = v37;
    if ( v37 <= 0x40 )
      return v36 == (char *)1;
    result = v19 - 1 == (unsigned int)sub_C444A0((__int64)&v36);
    goto LABEL_21;
  }
  if ( v29 != 2 )
  {
    if ( !v29 )
    {
      v17 = v37;
      LOBYTE(v32) = (_BYTE)v36;
      if ( v37 > 0x40 )
        v32 = *(_QWORD *)v36;
      result = v32 & 1;
LABEL_17:
      if ( v17 <= 0x40 )
        return result;
      goto LABEL_21;
    }
LABEL_53:
    BUG();
  }
  v30 = v37;
  result = 1;
  if ( !v37 )
    return result;
  if ( v37 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) == (_QWORD)v36;
  result = v30 == (unsigned int)sub_C445E0((__int64)&v36);
LABEL_21:
  if ( v36 )
  {
    v35 = result;
    j_j___libc_free_0_0((unsigned __int64)v36);
    return v35;
  }
  return result;
}
