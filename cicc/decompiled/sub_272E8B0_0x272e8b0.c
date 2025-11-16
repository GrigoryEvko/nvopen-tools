// Function: sub_272E8B0
// Address: 0x272e8b0
//
__int64 __fastcall sub_272E8B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  char **v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdi
  char **v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 v16; // r13
  __int64 v17; // r14
  char **v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // r14
  unsigned __int64 v21; // r13
  __int64 v22; // rbx
  char **v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v6 = a5;
  v8 = a3;
  v9 = a1;
  v28 = a4;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      v12 = *(_QWORD *)(v9 + 144);
      v13 = *(_QWORD *)(v8 + 144);
      if ( *(_QWORD *)(v13 + 8) == *(_QWORD *)(v12 + 8) )
      {
        if ( (int)sub_C49970(v13 + 24, (unsigned __int64 *)(v12 + 24)) < 0 )
        {
LABEL_4:
          v10 = (char **)v8;
          v11 = v6;
          v8 += 168;
          v6 += 168;
          sub_272D8A0(v11, v10, a3, a4, a5, a6);
          *(_QWORD *)(v6 - 24) = *(_QWORD *)(v8 - 24);
          *(_QWORD *)(v6 - 16) = *(_QWORD *)(v8 - 16);
          *(_DWORD *)(v6 - 8) = *(_DWORD *)(v8 - 8);
          if ( v9 == a2 )
            break;
          continue;
        }
      }
      else if ( *(_DWORD *)(v13 + 32) < *(_DWORD *)(v12 + 32) )
      {
        goto LABEL_4;
      }
      v14 = (char **)v9;
      v15 = v6;
      v9 += 168;
      v6 += 168;
      sub_272D8A0(v15, v14, a3, a4, a5, a6);
      *(_QWORD *)(v6 - 24) = *(_QWORD *)(v9 - 24);
      *(_QWORD *)(v6 - 16) = *(_QWORD *)(v9 - 16);
      *(_DWORD *)(v6 - 8) = *(_DWORD *)(v9 - 8);
      if ( v9 == a2 )
        break;
    }
    while ( v8 != v28 );
  }
  v27 = a2 - v9;
  v16 = 0xCF3CF3CF3CF3CF3DLL * ((a2 - v9) >> 3);
  if ( a2 - v9 > 0 )
  {
    v17 = v6;
    do
    {
      v18 = (char **)v9;
      v19 = v17;
      v9 += 168;
      v17 += 168;
      sub_272D8A0(v19, v18, a3, a4, a5, a6);
      *(_QWORD *)(v17 - 24) = *(_QWORD *)(v9 - 24);
      *(_QWORD *)(v17 - 16) = *(_QWORD *)(v9 - 16);
      a3 = *(unsigned int *)(v9 - 8);
      *(_DWORD *)(v17 - 8) = a3;
      --v16;
    }
    while ( v16 );
    a4 = v27;
    if ( v27 <= 0 )
      a4 = 168;
    v6 += a4;
  }
  v20 = v28 - v8;
  v21 = 0xCF3CF3CF3CF3CF3DLL * ((v28 - v8) >> 3);
  if ( v28 - v8 > 0 )
  {
    v22 = v6;
    do
    {
      v23 = (char **)v8;
      v24 = v22;
      v8 += 168;
      v22 += 168;
      sub_272D8A0(v24, v23, a3, a4, a5, a6);
      *(_QWORD *)(v22 - 24) = *(_QWORD *)(v8 - 24);
      *(_QWORD *)(v22 - 16) = *(_QWORD *)(v8 - 16);
      *(_DWORD *)(v22 - 8) = *(_DWORD *)(v8 - 8);
      --v21;
    }
    while ( v21 );
    v25 = 168;
    if ( v20 > 0 )
      v25 = v20;
    v6 += v25;
  }
  return v6;
}
