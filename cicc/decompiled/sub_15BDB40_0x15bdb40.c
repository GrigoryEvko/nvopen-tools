// Function: sub_15BDB40
// Address: 0x15bdb40
//
__int64 __fastcall sub_15BDB40(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        unsigned int a9,
        __int64 a10,
        unsigned int a11,
        __int64 a12,
        int a13,
        __int64 a14,
        __int64 a15,
        __int64 a16,
        unsigned __int64 a17,
        unsigned int a18,
        char a19)
{
  __int64 v23; // r10
  __int64 result; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r14
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r10
  __int64 v32; // rdi
  int v33; // [rsp+64h] [rbp-DCh]
  __int64 v34; // [rsp+68h] [rbp-D8h]
  __int64 v35; // [rsp+68h] [rbp-D8h]
  int v36; // [rsp+70h] [rbp-D0h]
  __int64 v37; // [rsp+78h] [rbp-C8h]
  __int64 v39; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+98h] [rbp-A8h] BYREF
  __int64 v41; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-98h] BYREF
  __int64 v43; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v44; // [rsp+B8h] [rbp-88h] BYREF
  __int64 v45; // [rsp+C0h] [rbp-80h]
  __int64 v46; // [rsp+C8h] [rbp-78h]
  unsigned __int64 v47; // [rsp+D0h] [rbp-70h]
  __int64 v48; // [rsp+D8h] [rbp-68h] BYREF
  int v49; // [rsp+E0h] [rbp-60h]
  __int64 v50; // [rsp+E8h] [rbp-58h]
  __int64 v51; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v52; // [rsp+F8h] [rbp-48h]
  __int64 v53; // [rsp+100h] [rbp-40h]

  if ( a18 )
  {
LABEL_4:
    v39 = a4;
    v40 = a6;
    v42 = a7;
    v41 = a3;
    v43 = a12;
    v44 = a14;
    v45 = a15;
    v46 = a16;
    v47 = a17;
    v25 = *a1 + 784;
    v26 = sub_161E980(56, 9);
    v27 = v26;
    if ( v26 )
    {
      sub_1623D80(v26, (_DWORD)a1, 13, a18, (unsigned int)&v39, 9, 0, 0);
      *(_WORD *)(v27 + 2) = a2;
      *(_DWORD *)(v27 + 24) = a5;
      *(_DWORD *)(v27 + 28) = a11;
      *(_QWORD *)(v27 + 32) = a8;
      *(_DWORD *)(v27 + 48) = a9;
      *(_QWORD *)(v27 + 40) = a10;
      *(_DWORD *)(v27 + 52) = a13;
    }
    return sub_15BDA60(v27, a18, v25);
  }
  v23 = *a1;
  v40 = a3;
  v41 = a4;
  LODWORD(v39) = a2;
  v43 = a6;
  LODWORD(v42) = a5;
  v44 = a7;
  v45 = a8;
  v46 = a10;
  v47 = __PAIR64__(a11, a9);
  v48 = a12;
  v49 = a13;
  v50 = a14;
  v51 = a15;
  v52 = a16;
  v53 = a17;
  v37 = *(_QWORD *)(v23 + 792);
  v34 = v23;
  v36 = *(_DWORD *)(v23 + 808);
  if ( !v36 )
    goto LABEL_3;
  v28 = (v36 - 1) & sub_15B5FF0(&v40, &v41, (int *)&v42, &v44, &v43, &v48, &v51);
  v29 = (__int64 *)(v37 + 8LL * v28);
  v30 = *v29;
  if ( *v29 == -8 )
    goto LABEL_3;
  v33 = 1;
  v31 = v34;
  while ( 1 )
  {
    if ( v30 != -16 && (_DWORD)v39 == *(unsigned __int16 *)(v30 + 2) )
    {
      v35 = *(unsigned int *)(v30 + 8);
      if ( v40 == *(_QWORD *)(v30 + 8 * (2 - v35)) )
      {
        v32 = v30;
        if ( *(_BYTE *)v30 != 15 )
          v32 = *(_QWORD *)(v30 - 8 * v35);
        if ( v41 == v32
          && (_DWORD)v42 == *(_DWORD *)(v30 + 24)
          && v43 == *(_QWORD *)(v30 + 8 * (1 - v35))
          && v44 == *(_QWORD *)(v30 + 8 * (3 - v35))
          && v45 == *(_QWORD *)(v30 + 32)
          && (_DWORD)v47 == *(_DWORD *)(v30 + 48)
          && v46 == *(_QWORD *)(v30 + 40)
          && HIDWORD(v47) == *(_DWORD *)(v30 + 28)
          && v48 == *(_QWORD *)(v30 + 8 * (4 - v35))
          && v49 == *(_DWORD *)(v30 + 52)
          && v50 == *(_QWORD *)(v30 + 8 * (5 - v35))
          && v51 == *(_QWORD *)(v30 + 8 * (6 - v35))
          && v52 == *(_QWORD *)(v30 + 8 * (7 - v35))
          && v53 == *(_QWORD *)(v30 + 8 * (8 - v35)) )
        {
          break;
        }
      }
    }
    v28 = (v36 - 1) & (v33 + v28);
    v29 = (__int64 *)(v37 + 8LL * v28);
    v30 = *v29;
    if ( *v29 == -8 )
      goto LABEL_3;
    ++v33;
  }
  if ( v29 == (__int64 *)(*(_QWORD *)(v31 + 792) + 8LL * *(unsigned int *)(v31 + 808)) || (result = *v29) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a19 )
      return result;
    goto LABEL_4;
  }
  return result;
}
