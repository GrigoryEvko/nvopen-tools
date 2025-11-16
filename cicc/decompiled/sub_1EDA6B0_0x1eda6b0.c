// Function: sub_1EDA6B0
// Address: 0x1eda6b0
//
__int64 __fastcall sub_1EDA6B0(__int64 *a1, __int64 a2)
{
  int v4; // r15d
  __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  unsigned int v9; // esi
  unsigned int v10; // ecx
  __int64 v11; // r10
  __int64 v12; // rbx
  __int64 v13; // r12
  __int16 v14; // ax
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rax
  int v18; // r8d
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  int v22; // r9d
  __int64 **v23; // rdi
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rsi
  _QWORD *v27; // rcx
  _QWORD *v28; // rdx
  __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  int v33; // [rsp+20h] [rbp-50h] BYREF
  int v34; // [rsp+24h] [rbp-4Ch] BYREF
  int v35; // [rsp+28h] [rbp-48h] BYREF
  int v36; // [rsp+2Ch] [rbp-44h] BYREF
  int v37; // [rsp+30h] [rbp-40h] BYREF
  int v38; // [rsp+34h] [rbp-3Ch] BYREF
  int v39; // [rsp+38h] [rbp-38h] BYREF
  int v40[13]; // [rsp+3Ch] [rbp-34h] BYREF

  sub_1ED87E0(a1[32], a2, &v35, &v33, &v36, &v34);
  v4 = v33;
  if ( v33 > 0 || v35 > 0 || !(unsigned __int8)sub_1ED88E0(v33, a2, a1[31]) )
    return 0;
  v8 = a1[34];
  v9 = v4 & 0x7FFFFFFF;
  v10 = *(_DWORD *)(v8 + 408);
  v11 = v4 & 0x7FFFFFFF;
  v32 = *(_QWORD *)(a2 + 24);
  if ( (v4 & 0x7FFFFFFFu) >= v10 || (v29 = *(_QWORD *)(*(_QWORD *)(v8 + 400) + 8LL * v9)) == 0 )
  {
    v16 = v9 + 1;
    if ( v10 < v9 + 1 )
    {
      v24 = v16;
      v25 = v10;
      if ( v16 < (unsigned __int64)v10 )
      {
        *(_DWORD *)(v8 + 408) = v16;
        v17 = *(_QWORD *)(v8 + 400);
        goto LABEL_24;
      }
      if ( v16 > (unsigned __int64)v10 )
      {
        if ( v16 > (unsigned __int64)*(unsigned int *)(v8 + 412) )
        {
          v31 = v16;
          sub_16CD150(v8 + 400, (const void *)(v8 + 416), v16, 8, v16, v16);
          v11 = v9;
          v16 = v9 + 1;
          v25 = *(unsigned int *)(v8 + 408);
          v24 = v31;
        }
        v17 = *(_QWORD *)(v8 + 400);
        v26 = *(_QWORD *)(v8 + 416);
        v27 = (_QWORD *)(v17 + 8 * v24);
        v28 = (_QWORD *)(v17 + 8 * v25);
        if ( v27 != v28 )
        {
          do
            *v28++ = v26;
          while ( v27 != v28 );
          v17 = *(_QWORD *)(v8 + 400);
        }
        *(_DWORD *)(v8 + 408) = v16;
        goto LABEL_24;
      }
    }
    v17 = *(_QWORD *)(v8 + 400);
LABEL_24:
    v30 = v11;
    *(_QWORD *)(v17 + 8LL * (v4 & 0x7FFFFFFF)) = sub_1DBA290(v4);
    v29 = *(_QWORD *)(*(_QWORD *)(v8 + 400) + 8 * v30);
    sub_1DBB110((_QWORD *)v8, v29);
    v6 = a1[31];
    v7 = (unsigned int)v35;
  }
  if ( (int)v7 < 0 )
    v12 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16 * (v7 & 0x7FFFFFFF) + 8);
  else
    v12 = *(_QWORD *)(*(_QWORD *)(v6 + 272) + 8 * v7);
  if ( !v12 )
    return 0;
  while ( (*(_BYTE *)(v12 + 4) & 8) != 0 )
  {
    v12 = *(_QWORD *)(v12 + 32);
    if ( !v12 )
      return 0;
  }
  v13 = *(_QWORD *)(v12 + 16);
LABEL_12:
  if ( a2 == v13 || (v14 = **(_WORD **)(v13 + 16), v14 != 15) && v14 != 10 || v32 != *(_QWORD *)(v13 + 24) )
  {
LABEL_15:
    v15 = v13;
    goto LABEL_17;
  }
  sub_1ED87E0(a1[32], a2, &v39, &v37, v40, &v38);
  v18 = v37;
  if ( v37 == v35 )
  {
    v18 = v39;
    v37 = v39;
  }
  if ( v18 > 0 || (unsigned __int8)sub_1ED88E0(v18, v13, a1[31]) )
  {
    v13 = *(_QWORD *)(v12 + 16);
    goto LABEL_15;
  }
  v23 = (__int64 **)sub_1E86160(a1[34], v21, v19, v20, v21, v22);
  if ( !*(_DWORD *)(v29 + 8) || (result = sub_1DB3D00(v23, v29, *(__int64 **)v29), !(_BYTE)result) )
  {
    v15 = *(_QWORD *)(v12 + 16);
LABEL_17:
    while ( 1 )
    {
      v12 = *(_QWORD *)(v12 + 32);
      if ( !v12 )
        return 0;
      if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
      {
        v13 = *(_QWORD *)(v12 + 16);
        if ( v13 != v15 )
          goto LABEL_12;
      }
    }
  }
  return result;
}
