// Function: sub_15C1830
// Address: 0x15c1830
//
__int64 __fastcall sub_15C1830(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        unsigned int a7,
        char a8)
{
  __int64 v8; // r10
  __int64 v12; // r9
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // r13
  int v18; // eax
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  int i; // [rsp+Ch] [rbp-94h]
  __int64 v25; // [rsp+20h] [rbp-80h]
  int v26; // [rsp+28h] [rbp-78h]
  __int64 v27; // [rsp+30h] [rbp-70h]
  __int64 v29; // [rsp+40h] [rbp-60h] BYREF
  __int64 v30; // [rsp+48h] [rbp-58h] BYREF
  __int64 v31; // [rsp+50h] [rbp-50h] BYREF
  __int64 v32; // [rsp+58h] [rbp-48h] BYREF
  int v33[16]; // [rsp+60h] [rbp-40h] BYREF

  v8 = a5;
  if ( a7 )
  {
LABEL_4:
    v14 = *a1;
    v30 = a3;
    v31 = a4;
    v29 = a2;
    v15 = v14 + 1328;
    v32 = v8;
    v16 = sub_161E980(32, 4);
    v17 = v16;
    if ( v16 )
    {
      sub_1623D80(v16, (_DWORD)a1, 31, a7, (unsigned int)&v29, 4, 0, 0);
      *(_WORD *)(v17 + 2) = 26;
      *(_DWORD *)(v17 + 24) = a6;
    }
    return sub_15C1670(v17, a7, v15);
  }
  v12 = *a1;
  v29 = a2;
  v30 = a3;
  v31 = a4;
  v32 = a5;
  v33[0] = a6;
  v25 = v12;
  v26 = *(_DWORD *)(v12 + 1352);
  v27 = *(_QWORD *)(v12 + 1336);
  if ( !v26 )
    goto LABEL_3;
  v18 = sub_15B3B60(&v29, &v30, &v31, &v32, v33);
  v8 = a5;
  v19 = (v26 - 1) & v18;
  v20 = (__int64 *)(v27 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == -8 )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v21 != -16 )
    {
      v22 = *(unsigned int *)(v21 + 8);
      if ( v29 == *(_QWORD *)(v21 - 8 * v22)
        && v30 == *(_QWORD *)(v21 + 8 * (1 - v22))
        && v31 == *(_QWORD *)(v21 + 8 * (2 - v22))
        && v32 == *(_QWORD *)(v21 + 8 * (3 - v22))
        && v33[0] == *(_DWORD *)(v21 + 24) )
      {
        break;
      }
    }
    v19 = (v26 - 1) & (i + v19);
    v20 = (__int64 *)(v27 + 8LL * v19);
    v21 = *v20;
    if ( *v20 == -8 )
      goto LABEL_3;
  }
  if ( v20 == (__int64 *)(*(_QWORD *)(v25 + 1336) + 8LL * *(unsigned int *)(v25 + 1352)) || (result = *v20) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a8 )
      return result;
    goto LABEL_4;
  }
  return result;
}
