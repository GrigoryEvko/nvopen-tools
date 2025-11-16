// Function: sub_15C2FB0
// Address: 0x15c2fb0
//
__int64 __fastcall sub_15C2FB0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        char a8,
        char a9,
        __int64 a10,
        int a11,
        unsigned int a12,
        char a13)
{
  __int64 v13; // r11
  __int64 v17; // r10
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // r14
  int v23; // eax
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  int i; // [rsp+2Ch] [rbp-B4h]
  __int64 v29; // [rsp+38h] [rbp-A8h]
  int v30; // [rsp+40h] [rbp-A0h]
  __int64 v31; // [rsp+48h] [rbp-98h]
  __int64 v33; // [rsp+60h] [rbp-80h] BYREF
  __int64 v34; // [rsp+68h] [rbp-78h] BYREF
  __int64 v35; // [rsp+70h] [rbp-70h] BYREF
  __int64 v36; // [rsp+78h] [rbp-68h] BYREF
  __int64 v37; // [rsp+80h] [rbp-60h] BYREF
  __int64 v38; // [rsp+88h] [rbp-58h] BYREF
  __int64 v39; // [rsp+90h] [rbp-50h] BYREF
  __int64 v40; // [rsp+98h] [rbp-48h] BYREF
  int v41; // [rsp+A0h] [rbp-40h]

  v13 = a2;
  if ( a12 )
  {
LABEL_4:
    v35 = a5;
    v38 = a4;
    v36 = a7;
    v33 = v13;
    v39 = a10;
    v19 = *a1;
    v34 = a3;
    v37 = a3;
    v20 = v19 + 1104;
    v21 = sub_161E980(40, 7);
    v22 = v21;
    if ( v21 )
    {
      sub_1623D80(v21, (_DWORD)a1, 24, a12, (unsigned int)&v33, 7, 0, 0);
      *(_WORD *)(v22 + 2) = 52;
      *(_DWORD *)(v22 + 24) = a6;
      *(_DWORD *)(v22 + 28) = a11;
      *(_BYTE *)(v22 + 32) = a8;
      *(_BYTE *)(v22 + 33) = a9;
    }
    return sub_15C2DA0(v22, a12, v20);
  }
  v17 = *a1;
  v33 = a2;
  v34 = a3;
  LODWORD(v37) = a6;
  v35 = a4;
  v38 = a7;
  v36 = a5;
  LOBYTE(v39) = a8;
  v29 = v17;
  BYTE1(v39) = a9;
  v40 = a10;
  v41 = a11;
  v31 = *(_QWORD *)(v17 + 1112);
  v30 = *(_DWORD *)(v17 + 1128);
  if ( !v30 )
    goto LABEL_3;
  v23 = sub_15B44C0(&v33, &v34, &v35, &v36, (int *)&v37, &v38, (__int8 *)&v39, (__int8 *)&v39 + 1, &v40);
  v13 = a2;
  v24 = (v30 - 1) & v23;
  v25 = (__int64 *)(v31 + 8LL * v24);
  v26 = *v25;
  if ( *v25 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v26 != -16 )
    {
      v27 = *(unsigned int *)(v26 + 8);
      if ( v33 == *(_QWORD *)(v26 - 8 * v27)
        && v34 == *(_QWORD *)(v26 + 8 * (1 - v27))
        && v35 == *(_QWORD *)(v26 + 8 * (5 - v27))
        && v36 == *(_QWORD *)(v26 + 8 * (2 - v27))
        && (_DWORD)v37 == *(_DWORD *)(v26 + 24)
        && v38 == *(_QWORD *)(v26 + 8 * (3 - v27))
        && (_WORD)v39 == *(_WORD *)(v26 + 32)
        && v40 == *(_QWORD *)(v26 + 8 * (6 - v27))
        && v41 == *(_DWORD *)(v26 + 28) )
      {
        break;
      }
    }
    v24 = (v30 - 1) & (i + v24);
    v25 = (__int64 *)(v31 + 8LL * v24);
    v26 = *v25;
    if ( *v25 == -8 )
      goto LABEL_3;
  }
  if ( v25 == (__int64 *)(*(_QWORD *)(v29 + 1112) + 8LL * *(unsigned int *)(v29 + 1128)) || (result = *v25) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a13 )
      return result;
    goto LABEL_4;
  }
  return result;
}
