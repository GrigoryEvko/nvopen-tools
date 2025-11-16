// Function: sub_162AE30
// Address: 0x162ae30
//
__int64 __fastcall sub_162AE30(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // r13d
  __int64 v6; // r12
  char v7; // al
  _QWORD *v8; // rdx
  __int64 result; // rax
  unsigned int v10; // esi
  __int64 *v11; // rdx
  __int64 v12; // rcx
  int i; // r10d
  unsigned int v14; // esi
  int v15; // eax
  int v16; // eax
  __int64 v17; // r14
  __int64 v18; // [rsp+8h] [rbp-118h]
  __int64 v19; // [rsp+68h] [rbp-B8h] BYREF
  _QWORD *v20; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v21; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v22; // [rsp+80h] [rbp-A0h] BYREF
  int v23; // [rsp+88h] [rbp-98h] BYREF
  __int64 v24; // [rsp+90h] [rbp-90h] BYREF
  __int64 v25; // [rsp+98h] [rbp-88h] BYREF
  __int64 v26; // [rsp+A0h] [rbp-80h]
  __int64 v27; // [rsp+A8h] [rbp-78h]
  int v28; // [rsp+B0h] [rbp-70h]
  int v29; // [rsp+B4h] [rbp-6Ch]
  __int64 v30; // [rsp+B8h] [rbp-68h] BYREF
  int v31; // [rsp+C0h] [rbp-60h]
  __int64 v32; // [rsp+C8h] [rbp-58h]
  __int64 v33; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v34; // [rsp+D8h] [rbp-48h]
  __int64 v35; // [rsp+E0h] [rbp-40h]

  v19 = a1;
  LODWORD(v20) = *(unsigned __int16 *)(a1 + 2);
  v3 = *(unsigned int *)(a1 + 8);
  v21 = *(_QWORD *)(a1 + 8 * (2 - v3));
  v4 = a1;
  if ( *(_BYTE *)a1 != 15 )
    v4 = *(_QWORD *)(a1 - 8LL * *(unsigned int *)(a1 + 8));
  v5 = *(_DWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v22 = v4;
  v23 = *(_DWORD *)(a1 + 24);
  v24 = *(_QWORD *)(a1 + 8 * (1 - v3));
  v25 = *(_QWORD *)(a1 + 8 * (3 - v3));
  v26 = *(_QWORD *)(a1 + 32);
  v27 = *(_QWORD *)(a1 + 40);
  v28 = *(_DWORD *)(a1 + 48);
  v29 = *(_DWORD *)(a1 + 28);
  v30 = *(_QWORD *)(a1 + 8 * (4 - v3));
  v31 = *(_DWORD *)(a1 + 52);
  v32 = *(_QWORD *)(a1 + 8 * (5 - v3));
  v33 = *(_QWORD *)(a1 + 8 * (6 - v3));
  v34 = *(_QWORD *)(a1 + 8 * (7 - v3));
  v35 = *(_QWORD *)(a1 + 8 * (8 - v3));
  if ( !v5 )
    goto LABEL_4;
  v10 = (v5 - 1) & sub_15B5FF0(&v21, &v22, &v23, &v25, &v24, &v30, &v33);
  v11 = (__int64 *)(v6 + 8LL * v10);
  v12 = *v11;
  if ( *v11 == -8 )
    goto LABEL_4;
  for ( i = 1; ; ++i )
  {
    if ( v12 != -16 && (_DWORD)v20 == *(unsigned __int16 *)(v12 + 2) )
    {
      v18 = *(unsigned int *)(v12 + 8);
      if ( v21 == *(_QWORD *)(v12 + 8 * (2 - v18)) )
      {
        v17 = v12;
        if ( *(_BYTE *)v12 != 15 )
          v17 = *(_QWORD *)(v12 - 8 * v18);
        if ( v22 == v17
          && v23 == *(_DWORD *)(v12 + 24)
          && v24 == *(_QWORD *)(v12 + 8 * (1 - v18))
          && v25 == *(_QWORD *)(v12 + 8 * (3 - v18))
          && v26 == *(_QWORD *)(v12 + 32)
          && v28 == *(_DWORD *)(v12 + 48)
          && v27 == *(_QWORD *)(v12 + 40)
          && v29 == *(_DWORD *)(v12 + 28)
          && v30 == *(_QWORD *)(v12 + 8 * (4 - v18))
          && v31 == *(_DWORD *)(v12 + 52)
          && v32 == *(_QWORD *)(v12 + 8 * (5 - v18))
          && v33 == *(_QWORD *)(v12 + 8 * (6 - v18))
          && v34 == *(_QWORD *)(v12 + 8 * (7 - v18))
          && v35 == *(_QWORD *)(v12 + 8 * (8 - v18)) )
        {
          break;
        }
      }
    }
    v10 = (v5 - 1) & (i + v10);
    v11 = (__int64 *)(v6 + 8LL * v10);
    v12 = *v11;
    if ( *v11 == -8 )
      goto LABEL_4;
  }
  if ( v11 == (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) || (result = *v11) == 0 )
  {
LABEL_4:
    v7 = sub_15B7D90(a2, &v19, &v20);
    v8 = v20;
    if ( v7 )
      return v19;
    v14 = *(_DWORD *)(a2 + 24);
    v15 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v16 = v15 + 1;
    if ( 4 * v16 >= 3 * v14 )
    {
      v14 *= 2;
    }
    else if ( v14 - *(_DWORD *)(a2 + 20) - v16 > v14 >> 3 )
    {
LABEL_14:
      *(_DWORD *)(a2 + 16) = v16;
      if ( *v8 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v8 = v19;
      return v19;
    }
    sub_15BD910(a2, v14);
    sub_15B7D90(a2, &v19, &v20);
    v8 = v20;
    v16 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_14;
  }
  return result;
}
