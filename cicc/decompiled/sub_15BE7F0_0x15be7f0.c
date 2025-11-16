// Function: sub_15BE7F0
// Address: 0x15be7f0
//
__int64 __fastcall sub_15BE7F0(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10,
        int a11,
        __int64 a12,
        unsigned int a13,
        char a14)
{
  __int64 v18; // r10
  __int64 result; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // r14
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r10
  __int64 v27; // rdi
  int v28; // [rsp+34h] [rbp-ACh]
  __int64 v29; // [rsp+38h] [rbp-A8h]
  __int64 v30; // [rsp+38h] [rbp-A8h]
  int v31; // [rsp+40h] [rbp-A0h]
  __int64 v32; // [rsp+48h] [rbp-98h]
  __int64 v34; // [rsp+60h] [rbp-80h] BYREF
  __int64 v35; // [rsp+68h] [rbp-78h] BYREF
  __int64 v36; // [rsp+70h] [rbp-70h] BYREF
  __int64 v37; // [rsp+78h] [rbp-68h] BYREF
  __int64 v38; // [rsp+80h] [rbp-60h] BYREF
  __int64 v39; // [rsp+88h] [rbp-58h] BYREF
  __int64 v40; // [rsp+90h] [rbp-50h]
  __int64 v41; // [rsp+98h] [rbp-48h]
  int v42; // [rsp+A0h] [rbp-40h]
  int v43; // [rsp+A4h] [rbp-3Ch]
  __int64 v44[7]; // [rsp+A8h] [rbp-38h] BYREF

  if ( a13 )
  {
LABEL_4:
    v34 = a4;
    v35 = a6;
    v37 = a7;
    v36 = a3;
    v38 = a12;
    v20 = *a1 + 1392;
    v21 = sub_161E980(56, 5);
    v22 = v21;
    if ( v21 )
    {
      sub_1623D80(v21, (_DWORD)a1, 33, a13, (unsigned int)&v34, 5, 0, 0);
      *(_WORD *)(v22 + 2) = a2;
      *(_DWORD *)(v22 + 24) = a5;
      *(_DWORD *)(v22 + 28) = a11;
      *(_QWORD *)(v22 + 32) = a8;
      *(_DWORD *)(v22 + 48) = a9;
      *(_QWORD *)(v22 + 40) = a10;
    }
    return sub_15BE5F0(v22, a13, v20);
  }
  v18 = *a1;
  v35 = a3;
  v36 = a4;
  LODWORD(v34) = a2;
  v38 = a6;
  LODWORD(v37) = a5;
  v29 = v18;
  v39 = a7;
  v40 = a8;
  v41 = a10;
  v42 = a9;
  v43 = a11;
  v44[0] = a12;
  v31 = *(_DWORD *)(v18 + 1416);
  v32 = *(_QWORD *)(v18 + 1400);
  if ( !v31 )
    goto LABEL_3;
  v23 = (v31 - 1) & sub_15B5D10(&v35, &v36, (int *)&v37, &v39, &v38, v44);
  v24 = (__int64 *)(v32 + 8LL * v23);
  v25 = *v24;
  if ( *v24 == -8 )
    goto LABEL_3;
  v28 = 1;
  v26 = v29;
  while ( 1 )
  {
    if ( v25 != -16 && (_DWORD)v34 == *(unsigned __int16 *)(v25 + 2) )
    {
      v30 = *(unsigned int *)(v25 + 8);
      if ( v35 == *(_QWORD *)(v25 + 8 * (2 - v30)) )
      {
        v27 = v25;
        if ( *(_BYTE *)v25 != 15 )
          v27 = *(_QWORD *)(v25 - 8 * v30);
        if ( v36 == v27
          && (_DWORD)v37 == *(_DWORD *)(v25 + 24)
          && v38 == *(_QWORD *)(v25 + 8 * (1 - v30))
          && v39 == *(_QWORD *)(v25 + 8 * (3 - v30))
          && v40 == *(_QWORD *)(v25 + 32)
          && v42 == *(_DWORD *)(v25 + 48)
          && v41 == *(_QWORD *)(v25 + 40)
          && v43 == *(_DWORD *)(v25 + 28)
          && v44[0] == *(_QWORD *)(v25 + 8 * (4 - v30)) )
        {
          break;
        }
      }
    }
    v23 = (v31 - 1) & (v28 + v23);
    v24 = (__int64 *)(v32 + 8LL * v23);
    v25 = *v24;
    if ( *v24 == -8 )
      goto LABEL_3;
    ++v28;
  }
  if ( v24 == (__int64 *)(*(_QWORD *)(v26 + 1400) + 8LL * *(unsigned int *)(v26 + 1416)) || (result = *v24) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a14 )
      return result;
    goto LABEL_4;
  }
  return result;
}
