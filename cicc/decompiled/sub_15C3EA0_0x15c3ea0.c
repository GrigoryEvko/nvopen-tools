// Function: sub_15C3EA0
// Address: 0x15c3ea0
//
__int64 __fastcall sub_15C3EA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6, char a7)
{
  char v12; // r9
  __int64 v13; // r8
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r14
  int v19; // r10d
  unsigned int v20; // r10d
  __int64 *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdi
  int i; // [rsp+10h] [rbp-80h]
  __int64 v25; // [rsp+20h] [rbp-70h]
  __int64 v26; // [rsp+28h] [rbp-68h]
  int v27; // [rsp+30h] [rbp-60h]
  __int64 v28; // [rsp+38h] [rbp-58h]
  __int64 v29; // [rsp+40h] [rbp-50h] BYREF
  __int64 v30; // [rsp+48h] [rbp-48h] BYREF
  __int64 v31; // [rsp+50h] [rbp-40h]
  int v32[14]; // [rsp+58h] [rbp-38h] BYREF

  v12 = a7;
  if ( a6 )
  {
LABEL_4:
    v15 = *a1;
    v29 = a2;
    v30 = a3;
    v31 = a4;
    v16 = v15 + 1168;
    v17 = sub_161E980(32, 3);
    v18 = v17;
    if ( v17 )
    {
      sub_1623D80(v17, (_DWORD)a1, 26, a6, (unsigned int)&v29, 3, 0, 0);
      *(_DWORD *)(v18 + 24) = a5;
      *(_WORD *)(v18 + 2) = 10;
    }
    return sub_15C3CF0(v18, a6, v16);
  }
  v13 = *a1;
  v29 = a2;
  v30 = a3;
  v31 = a4;
  v32[0] = a5;
  v26 = v13;
  v27 = *(_DWORD *)(v13 + 1192);
  v28 = *(_QWORD *)(v13 + 1176);
  if ( !v27 )
    goto LABEL_3;
  v25 = a4;
  v19 = sub_15B3EF0(&v29, &v30, v32);
  a4 = v25;
  v12 = a7;
  v20 = (v27 - 1) & v19;
  v21 = (__int64 *)(v28 + 8LL * v20);
  v22 = *v21;
  if ( *v21 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v22 != -16 )
    {
      v23 = *(unsigned int *)(v22 + 8);
      if ( v29 == *(_QWORD *)(v22 - 8 * v23)
        && v30 == *(_QWORD *)(v22 + 8 * (1 - v23))
        && v31 == *(_QWORD *)(v22 + 8 * (2 - v23))
        && v32[0] == *(_DWORD *)(v22 + 24) )
      {
        break;
      }
    }
    v20 = (v27 - 1) & (i + v20);
    v21 = (__int64 *)(v28 + 8LL * v20);
    v22 = *v21;
    if ( *v21 == -8 )
      goto LABEL_3;
  }
  if ( v21 == (__int64 *)(*(_QWORD *)(v26 + 1176) + 8LL * *(unsigned int *)(v26 + 1192)) || (result = *v21) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !v12 )
      return result;
    goto LABEL_4;
  }
  return result;
}
