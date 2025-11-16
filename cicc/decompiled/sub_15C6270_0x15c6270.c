// Function: sub_15C6270
// Address: 0x15c6270
//
__int64 __fastcall sub_15C6270(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        unsigned int a8,
        char a9)
{
  __int64 v12; // r10
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r14
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r10
  __int64 v22; // [rsp+18h] [rbp-88h]
  __int64 v23; // [rsp+18h] [rbp-88h]
  int v24; // [rsp+20h] [rbp-80h]
  __int64 v25; // [rsp+28h] [rbp-78h]
  int v26; // [rsp+30h] [rbp-70h]
  __int64 v28; // [rsp+40h] [rbp-60h] BYREF
  __int64 v29; // [rsp+48h] [rbp-58h] BYREF
  __int64 v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+58h] [rbp-48h] BYREF
  int v32; // [rsp+60h] [rbp-40h] BYREF
  __int64 v33[7]; // [rsp+68h] [rbp-38h] BYREF

  if ( a8 )
  {
LABEL_4:
    v29 = a4;
    v31 = a5;
    v30 = a7;
    v14 = *a1;
    v28 = a3;
    v15 = v14 + 1232;
    v16 = sub_161E980(32, 4);
    v17 = v16;
    if ( v16 )
    {
      sub_1623D80(v16, (_DWORD)a1, 28, a8, (unsigned int)&v28, 4, 0, 0);
      *(_WORD *)(v17 + 2) = a2;
      *(_DWORD *)(v17 + 24) = a6;
    }
    return sub_15C60A0(v17, a8, v15);
  }
  v12 = *a1;
  v29 = a3;
  v30 = a4;
  LODWORD(v28) = a2;
  v31 = a5;
  v32 = a6;
  v22 = v12;
  v33[0] = a7;
  v24 = *(_DWORD *)(v12 + 1256);
  v25 = *(_QWORD *)(v12 + 1240);
  if ( !v24 )
    goto LABEL_3;
  v18 = (v24 - 1) & sub_15B6E40((int *)&v28, &v29, &v30, &v31, &v32, v33);
  v19 = (__int64 *)(v25 + 8LL * v18);
  v20 = *v19;
  if ( *v19 == -8 )
    goto LABEL_3;
  v26 = 1;
  v21 = v22;
  while ( 1 )
  {
    if ( v20 != -16 && (_DWORD)v28 == *(unsigned __int16 *)(v20 + 2) )
    {
      v23 = *(unsigned int *)(v20 + 8);
      if ( v29 == *(_QWORD *)(v20 - 8 * v23)
        && v30 == *(_QWORD *)(v20 + 8 * (1 - v23))
        && v31 == *(_QWORD *)(v20 + 8 * (3 - v23))
        && v32 == *(_DWORD *)(v20 + 24)
        && v33[0] == *(_QWORD *)(v20 + 8 * (2 - v23)) )
      {
        break;
      }
    }
    v18 = (v24 - 1) & (v26 + v18);
    v19 = (__int64 *)(v25 + 8LL * v18);
    v20 = *v19;
    if ( *v19 == -8 )
      goto LABEL_3;
    ++v26;
  }
  if ( v19 == (__int64 *)(*(_QWORD *)(v21 + 1240) + 8LL * *(unsigned int *)(v21 + 1256)) || (result = *v19) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a9 )
      return result;
    goto LABEL_4;
  }
  return result;
}
