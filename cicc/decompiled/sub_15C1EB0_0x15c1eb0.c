// Function: sub_15C1EB0
// Address: 0x15c1eb0
//
__int64 __fastcall sub_15C1EB0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        char a8)
{
  __int64 v8; // r11
  __int64 v12; // r10
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  int v18; // eax
  unsigned int v19; // ecx
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // [rsp+20h] [rbp-90h]
  __int64 v24; // [rsp+30h] [rbp-80h]
  int v25; // [rsp+38h] [rbp-78h]
  __int64 v26; // [rsp+40h] [rbp-70h]
  int i; // [rsp+48h] [rbp-68h]
  __int64 v28; // [rsp+50h] [rbp-60h] BYREF
  __int64 v29; // [rsp+58h] [rbp-58h] BYREF
  __int64 v30; // [rsp+60h] [rbp-50h] BYREF
  __int64 v31; // [rsp+68h] [rbp-48h] BYREF
  __int64 v32[8]; // [rsp+70h] [rbp-40h] BYREF

  v8 = a2;
  if ( a7 )
  {
LABEL_4:
    v14 = *a1;
    v30 = a4;
    v31 = a5;
    v28 = v8;
    v15 = v14 + 1008;
    v29 = a3;
    v32[0] = a6;
    v16 = sub_161E980(24, 5);
    v17 = v16;
    if ( v16 )
    {
      sub_1623D80(v16, (_DWORD)a1, 21, a7, (unsigned int)&v28, 5, 0, 0);
      *(_WORD *)(v17 + 2) = 30;
    }
    return sub_15C1CE0(v17, a7, v15);
  }
  v12 = *a1;
  v28 = a2;
  v29 = a3;
  v30 = a4;
  v31 = a5;
  v32[0] = a6;
  v24 = v12;
  v26 = *(_QWORD *)(v12 + 1016);
  v25 = *(_DWORD *)(v12 + 1032);
  if ( !v25 )
    goto LABEL_3;
  v23 = a6;
  v18 = sub_15B6BA0(&v28, &v29, &v30, &v31, v32);
  v8 = a2;
  a6 = v23;
  v19 = (v25 - 1) & v18;
  v20 = (__int64 *)(v26 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v21 != -16 )
    {
      v22 = *(unsigned int *)(v21 + 8);
      if ( v28 == *(_QWORD *)(v21 - 8 * v22)
        && v29 == *(_QWORD *)(v21 + 8 * (1 - v22))
        && v30 == *(_QWORD *)(v21 + 8 * (2 - v22))
        && v31 == *(_QWORD *)(v21 + 8 * (3 - v22))
        && v32[0] == *(_QWORD *)(v21 + 8 * (4 - v22)) )
      {
        break;
      }
    }
    v19 = (v25 - 1) & (i + v19);
    v20 = (__int64 *)(v26 + 8LL * v19);
    v21 = *v20;
    if ( *v20 == -8 )
      goto LABEL_3;
  }
  if ( v20 == (__int64 *)(*(_QWORD *)(v24 + 1016) + 8LL * *(unsigned int *)(v24 + 1032)) || (result = *v20) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a8 )
      return result;
    goto LABEL_4;
  }
  return result;
}
