// Function: sub_15C37C0
// Address: 0x15c37c0
//
__int64 __fastcall sub_15C37C0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        int a7,
        int a8,
        int a9,
        unsigned int a10,
        char a11)
{
  __int64 v11; // r11
  __int64 v15; // r10
  __int64 result; // rax
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r14
  int v21; // eax
  unsigned int v22; // ecx
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rsi
  int i; // [rsp+20h] [rbp-A0h]
  __int64 v27; // [rsp+30h] [rbp-90h]
  int v28; // [rsp+38h] [rbp-88h]
  __int64 v29; // [rsp+40h] [rbp-80h]
  __int64 v31; // [rsp+50h] [rbp-70h] BYREF
  __int64 v32; // [rsp+58h] [rbp-68h] BYREF
  __int64 v33; // [rsp+60h] [rbp-60h] BYREF
  __int64 v34; // [rsp+68h] [rbp-58h] BYREF
  __int64 v35; // [rsp+70h] [rbp-50h] BYREF
  int v36; // [rsp+78h] [rbp-48h] BYREF
  int v37; // [rsp+7Ch] [rbp-44h] BYREF
  int v38; // [rsp+80h] [rbp-40h]

  v11 = a2;
  if ( a10 )
  {
LABEL_4:
    v17 = *a1;
    v33 = a4;
    v34 = a6;
    v31 = v11;
    v18 = v17 + 1136;
    v32 = a3;
    v19 = sub_161E980(40, 4);
    v20 = v19;
    if ( v19 )
    {
      sub_1623D80(v19, (_DWORD)a1, 25, a10, (unsigned int)&v31, 4, 0, 0);
      *(_WORD *)(v20 + 2) = 52;
      *(_DWORD *)(v20 + 24) = a5;
      *(_DWORD *)(v20 + 28) = a9;
      *(_WORD *)(v20 + 32) = a7;
      *(_DWORD *)(v20 + 36) = a8;
    }
    return sub_15C35E0(v20, a10, v18);
  }
  v15 = *a1;
  v31 = a2;
  v32 = a3;
  LODWORD(v34) = a5;
  v33 = a4;
  v36 = a7;
  v35 = a6;
  v37 = a8;
  v27 = v15;
  v38 = a9;
  v29 = *(_QWORD *)(v15 + 1144);
  v28 = *(_DWORD *)(v15 + 1160);
  if ( !v28 )
    goto LABEL_3;
  v21 = sub_15B41C0(&v31, &v32, &v33, (int *)&v34, &v35, &v36, &v37);
  v11 = a2;
  v22 = (v28 - 1) & v21;
  v23 = (__int64 *)(v29 + 8LL * v22);
  v24 = *v23;
  if ( *v23 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v24 != -16 )
    {
      v25 = *(unsigned int *)(v24 + 8);
      if ( v31 == *(_QWORD *)(v24 - 8 * v25)
        && v32 == *(_QWORD *)(v24 + 8 * (1 - v25))
        && v33 == *(_QWORD *)(v24 + 8 * (2 - v25))
        && (_DWORD)v34 == *(_DWORD *)(v24 + 24)
        && v35 == *(_QWORD *)(v24 + 8 * (3 - v25))
        && v36 == *(unsigned __int16 *)(v24 + 32)
        && v37 == *(_DWORD *)(v24 + 36)
        && v38 == *(_DWORD *)(v24 + 28) )
      {
        break;
      }
    }
    v22 = (v28 - 1) & (i + v22);
    v23 = (__int64 *)(v29 + 8LL * v22);
    v24 = *v23;
    if ( *v23 == -8 )
      goto LABEL_3;
  }
  if ( v23 == (__int64 *)(*(_QWORD *)(v27 + 1144) + 8LL * *(unsigned int *)(v27 + 1160)) || (result = *v23) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a11 )
      return result;
    goto LABEL_4;
  }
  return result;
}
