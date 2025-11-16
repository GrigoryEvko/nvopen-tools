// Function: sub_2D54A80
// Address: 0x2d54a80
//
__int64 *__fastcall sub_2D54A80(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  _BYTE *v7; // rdx
  unsigned __int64 *v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rdx
  char *v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  const char *v15; // rax
  __int128 v16; // [rsp-30h] [rbp-D0h]
  __int128 v17; // [rsp-20h] [rbp-C0h]
  __int64 v18; // [rsp-10h] [rbp-B0h]
  unsigned __int64 v19[5]; // [rsp+8h] [rbp-98h] BYREF
  __int64 v20; // [rsp+30h] [rbp-70h]
  _QWORD v21[2]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 *v22; // [rsp+50h] [rbp-50h]
  __int64 v23; // [rsp+58h] [rbp-48h]
  __int64 v24; // [rsp+60h] [rbp-40h]

  v6 = a2[8];
  v7 = (_BYTE *)a2[7];
  v19[0] = 0;
  if ( !v6 || *v7 != 118 )
    goto LABEL_3;
  v9 = (unsigned __int64 *)(v7 + 1);
  v10 = v6 - 1;
  if ( sub_C93C90((__int64)(v7 + 1), v6 - 1, 0xAu, v19) )
  {
    v22 = v9;
    LOWORD(v24) = 1283;
    v12 = "'";
    v19[3] = (unsigned __int64)"'";
    LOWORD(v20) = 770;
    v23 = v10;
    v18 = v20;
    *((_QWORD *)&v17 + 1) = v19[4];
    *(_QWORD *)&v17 = "'";
    *((_QWORD *)&v16 + 1) = v19[2];
    v21[0] = "version number expected: '";
    v15 = (const char *)v21;
    goto LABEL_12;
  }
  if ( v19[0] > 1 )
  {
    v15 = "invalid profile version: ";
    v11 = 3331;
    v22 = v19;
    LOWORD(v24) = 3331;
    v21[0] = "invalid profile version: ";
    v18 = v24;
    *((_QWORD *)&v17 + 1) = v23;
    *(_QWORD *)&v17 = v19;
    *((_QWORD *)&v16 + 1) = v21[1];
LABEL_12:
    *(_QWORD *)&v16 = v15;
    sub_2D507F0(a1, a2, v11, (__int64)v12, v13, v14, v16, v17, v18);
    return a1;
  }
  sub_C7C5C0((__int64)(a2 + 1));
  if ( !v19[0] )
  {
LABEL_3:
    sub_2D537F0(a1, (__int64)a2, (__int64)v7, a4, a5, a6);
    return a1;
  }
  if ( v19[0] != 1 )
    BUG();
  sub_2D520E0(a1, (__int64)a2);
  return a1;
}
