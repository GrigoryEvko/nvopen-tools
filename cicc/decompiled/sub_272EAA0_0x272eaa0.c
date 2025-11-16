// Function: sub_272EAA0
// Address: 0x272eaa0
//
__int64 __fastcall sub_272EAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v9; // r12
  char **v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdi
  char **v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // r15
  unsigned __int64 v17; // r13
  __int64 v18; // r12
  char **v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rax
  unsigned __int64 v23; // r13
  __int64 v24; // r15
  char **v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v6 = a5;
  v7 = a3;
  v28 = a4;
  if ( a2 == a1 )
    goto LABEL_8;
  v9 = a1;
  while ( v28 != v7 )
  {
    v12 = *(_QWORD *)(v9 + 144);
    v13 = *(_QWORD *)(v7 + 144);
    if ( *(_QWORD *)(v13 + 8) == *(_QWORD *)(v12 + 8) )
    {
      if ( (int)sub_C49970(v13 + 24, (unsigned __int64 *)(v12 + 24)) >= 0 )
        goto LABEL_7;
LABEL_3:
      v10 = (char **)v7;
      v11 = v6;
      v7 += 168;
      v6 += 168;
      sub_272D8A0(v11, v10, a3, a4, a5, a6);
      *(_QWORD *)(v6 - 24) = *(_QWORD *)(v7 - 24);
      *(_QWORD *)(v6 - 16) = *(_QWORD *)(v7 - 16);
      *(_DWORD *)(v6 - 8) = *(_DWORD *)(v7 - 8);
      if ( a2 == v9 )
        goto LABEL_8;
    }
    else
    {
      if ( *(_DWORD *)(v13 + 32) < *(_DWORD *)(v12 + 32) )
        goto LABEL_3;
LABEL_7:
      v14 = (char **)v9;
      v15 = v6;
      v9 += 168;
      v6 += 168;
      sub_272D8A0(v15, v14, a3, a4, a5, a6);
      *(_QWORD *)(v6 - 24) = *(_QWORD *)(v9 - 24);
      *(_QWORD *)(v6 - 16) = *(_QWORD *)(v9 - 16);
      *(_DWORD *)(v6 - 8) = *(_DWORD *)(v9 - 8);
      if ( a2 == v9 )
        goto LABEL_8;
    }
  }
  v27 = a2 - v9;
  v23 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - v9) >> 3);
  if ( a2 - v9 <= 0 )
    return v6;
  v24 = v6;
  do
  {
    v25 = (char **)v9;
    v26 = v24;
    v9 += 168;
    v24 += 168;
    sub_272D8A0(v26, v25, a3, a4, a5, a6);
    *(_QWORD *)(v24 - 24) = *(_QWORD *)(v9 - 24);
    *(_QWORD *)(v24 - 16) = *(_QWORD *)(v9 - 16);
    a3 = *(unsigned int *)(v9 - 8);
    *(_DWORD *)(v24 - 8) = a3;
    --v23;
  }
  while ( v23 );
  a4 = v27;
  if ( v27 <= 0 )
    a4 = 168;
  v6 += a4;
LABEL_8:
  v16 = v28 - v7;
  v17 = 0xCF3CF3CF3CF3CF3DLL * ((v28 - v7) >> 3);
  if ( v28 - v7 <= 0 )
    return v6;
  v18 = v6;
  do
  {
    v19 = (char **)v7;
    v20 = v18;
    v7 += 168;
    v18 += 168;
    sub_272D8A0(v20, v19, a3, a4, a5, a6);
    *(_QWORD *)(v18 - 24) = *(_QWORD *)(v7 - 24);
    *(_QWORD *)(v18 - 16) = *(_QWORD *)(v7 - 16);
    *(_DWORD *)(v18 - 8) = *(_DWORD *)(v7 - 8);
    --v17;
  }
  while ( v17 );
  v21 = 168;
  if ( v16 > 0 )
    v21 = v16;
  return v6 + v21;
}
