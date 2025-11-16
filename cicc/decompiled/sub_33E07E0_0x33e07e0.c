// Function: sub_33E07E0
// Address: 0x33e07e0
//
bool __fastcall sub_33E07E0(__int64 a1, unsigned __int64 a2, unsigned __int8 a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  unsigned int v7; // edx
  __int64 v8; // rsi
  unsigned __int16 *v9; // rdx
  int v10; // eax
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdi
  unsigned int v18; // r12d
  int v19; // r8d
  bool result; // al
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  unsigned __int16 *v24; // rdx
  unsigned __int16 v25; // ax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int16 v29; // [rsp+10h] [rbp-70h] BYREF
  __int64 v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  __int64 v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h] BYREF
  __int64 v36; // [rsp+48h] [rbp-38h]

  v6 = sub_33CF5B0(a1, a2);
  v8 = v7;
  v9 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * v7);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v29 = v10;
  v30 = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    {
      LOWORD(v35) = v10;
      v36 = v11;
      goto LABEL_4;
    }
    LOWORD(v10) = word_4456580[v10 - 1];
    v13 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v29) )
    {
      v36 = v11;
      LOWORD(v35) = 0;
      goto LABEL_9;
    }
    LOWORD(v10) = sub_3009970((__int64)&v29, v8, v21, v22, v23);
  }
  LOWORD(v35) = v10;
  v36 = v13;
  if ( !(_WORD)v10 )
  {
LABEL_9:
    v31 = sub_3007260((__int64)&v35);
    LODWORD(v12) = v31;
    v32 = v14;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
    goto LABEL_26;
  v12 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
LABEL_10:
  v15 = sub_33DFBC0(v6, v8, a3, 0, v4, v5);
  v16 = v15;
  if ( !v15 )
    return 0;
  v17 = *(_QWORD *)(v15 + 96);
  v18 = *(_DWORD *)(v17 + 32);
  if ( !v18 )
  {
LABEL_19:
    v24 = *(unsigned __int16 **)(v16 + 48);
    v25 = *v24;
    v26 = *((_QWORD *)v24 + 1);
    LOWORD(v33) = v25;
    v34 = v26;
    if ( !v25 )
    {
      v27 = sub_3007260((__int64)&v33);
      v35 = v27;
      v36 = v28;
LABEL_21:
      v33 = v27;
      LOBYTE(v34) = v28;
      return (unsigned int)v12 == sub_CA1930(&v33);
    }
    if ( v25 != 1 && (unsigned __int16)(v25 - 504) > 7u )
    {
      v28 = 16LL * (v25 - 1);
      v27 = *(_QWORD *)&byte_444C4A0[v28];
      LOBYTE(v28) = byte_444C4A0[v28 + 8];
      goto LABEL_21;
    }
LABEL_26:
    BUG();
  }
  if ( v18 <= 0x40 )
  {
    if ( *(_QWORD *)(v17 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) )
      goto LABEL_19;
    return 0;
  }
  v19 = sub_C445E0(v17 + 24);
  result = 0;
  if ( v18 == v19 )
    goto LABEL_19;
  return result;
}
