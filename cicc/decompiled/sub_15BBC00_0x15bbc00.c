// Function: sub_15BBC00
// Address: 0x15bbc00
//
__int64 __fastcall sub_15BBC00(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9,
        char a10)
{
  __int64 v12; // r10
  __int64 result; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r14
  int v17; // r11d
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r10
  __int64 v22; // [rsp+28h] [rbp-98h]
  __int64 v23; // [rsp+28h] [rbp-98h]
  unsigned int v24; // [rsp+30h] [rbp-90h]
  int v25; // [rsp+34h] [rbp-8Ch]
  int v26; // [rsp+34h] [rbp-8Ch]
  __int64 v27; // [rsp+38h] [rbp-88h]
  __int64 v30; // [rsp+50h] [rbp-70h] BYREF
  __int64 v31; // [rsp+58h] [rbp-68h] BYREF
  __int64 v32; // [rsp+60h] [rbp-60h] BYREF
  __int64 v33; // [rsp+68h] [rbp-58h] BYREF
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF
  __int64 v35; // [rsp+78h] [rbp-48h] BYREF
  __int64 v36[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( a9 )
  {
LABEL_4:
    v30 = a5;
    v31 = a6;
    v32 = a7;
    v33 = a8;
    v14 = *a1 + 1424;
    v15 = sub_161E980(48, 4);
    v16 = v15;
    if ( v15 )
    {
      sub_1623D80(v15, (_DWORD)a1, 34, a9, (unsigned int)&v30, 4, 0, 0);
      *(_QWORD *)(v16 + 24) = a2;
      *(_WORD *)(v16 + 2) = 33;
      *(_QWORD *)(v16 + 32) = a3;
      *(_BYTE *)(v16 + 40) = a4;
    }
    return sub_15BBA20(v16, a9, v14);
  }
  v12 = *a1;
  v30 = a2;
  v33 = a5;
  v31 = a3;
  v34 = a6;
  LOBYTE(v32) = a4;
  v22 = v12;
  v35 = a7;
  v36[0] = a8;
  v25 = *(_DWORD *)(v12 + 1448);
  v27 = *(_QWORD *)(v12 + 1432);
  if ( !v25 )
    goto LABEL_3;
  v18 = (v25 - 1) & sub_15B3480(&v30, &v31, (char *)&v32, &v35, v36, &v33, &v34);
  v19 = (__int64 *)(v27 + 8LL * v18);
  v20 = *v19;
  if ( *v19 == -8 )
    goto LABEL_3;
  v24 = v18;
  v17 = v25 - 1;
  v26 = 1;
  v21 = v22;
  while ( 1 )
  {
    if ( v20 != -16
      && v30 == *(_QWORD *)(v20 + 24)
      && v31 == *(_QWORD *)(v20 + 32)
      && (_BYTE)v32 == *(_BYTE *)(v20 + 40) )
    {
      v23 = *(unsigned int *)(v20 + 8);
      if ( v33 == *(_QWORD *)(v20 - 8 * v23)
        && v34 == *(_QWORD *)(v20 + 8 * (1 - v23))
        && v35 == *(_QWORD *)(v20 + 8 * (2 - v23))
        && v36[0] == *(_QWORD *)(v20 + 8 * (3 - v23)) )
      {
        break;
      }
    }
    v24 = v17 & (v26 + v24);
    v19 = (__int64 *)(v27 + 8LL * v24);
    v20 = *v19;
    if ( *v19 == -8 )
      goto LABEL_3;
    ++v26;
  }
  if ( v19 == (__int64 *)(*(_QWORD *)(v21 + 1432) + 8LL * *(unsigned int *)(v21 + 1448)) || (result = *v19) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a10 )
      return result;
    goto LABEL_4;
  }
  return result;
}
