// Function: sub_14AC3D0
// Address: 0x14ac3d0
//
__int64 __fastcall sub_14AC3D0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned int v8; // r13d
  __int64 v9; // rbx
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v14; // r13
  __int64 v15; // r12
  int v16; // eax
  __int64 result; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // r14
  __int64 v22; // r14
  __int64 v23; // r13
  _QWORD *v24; // r12
  _QWORD *v25; // rbx
  _QWORD *v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-80h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  _QWORD *v30; // [rsp+18h] [rbp-68h]
  __int64 v31; // [rsp+28h] [rbp-58h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+28h] [rbp-58h]
  char *v34; // [rsp+30h] [rbp-50h] BYREF
  char v35; // [rsp+40h] [rbp-40h]
  char v36; // [rsp+41h] [rbp-3Fh]

  v6 = a6;
  v7 = a1;
  v8 = a5;
  v9 = a4;
  v30 = a2;
  if ( *(_BYTE *)(a3 + 8) == 13 )
  {
    if ( *(_DWORD *)(a3 + 12) )
    {
      v29 = *(unsigned int *)(a3 + 12);
      v28 = a4 + 16;
      v11 = *(unsigned int *)(a4 + 8);
      v12 = a3;
      v14 = (__int64)a2;
      v15 = 0;
      v16 = 0;
      if ( (unsigned int)v11 >= *(_DWORD *)(a4 + 12) )
        goto LABEL_7;
      while ( 1 )
      {
        v31 = a6;
        *(_DWORD *)(*(_QWORD *)v9 + 4 * v11) = v16;
        ++*(_DWORD *)(v9 + 8);
        result = sub_14AC3D0(a1, v14, *(_QWORD *)(*(_QWORD *)(v12 + 16) + 8 * v15), v9, a5, a6);
        v19 = *(unsigned int *)(v9 + 8);
        a6 = v31;
        v11 = (unsigned int)(v19 - 1);
        *(_DWORD *)(v9 + 8) = v11;
        if ( !result )
          break;
        if ( v29 == ++v15 )
          return result;
        v14 = result;
        v16 = v15;
        if ( (unsigned int)v11 >= *(_DWORD *)(v9 + 12) )
        {
LABEL_7:
          v27 = a6;
          sub_16CD150(v9, v28, 0, 4);
          v11 = *(unsigned int *)(v9 + 8);
          a6 = v27;
          v16 = v15;
        }
      }
      v24 = (_QWORD *)v14;
      v7 = a1;
      v8 = a5;
      v6 = v31;
      if ( v30 == v24 )
      {
        v30 = 0;
      }
      else
      {
        v33 = v9;
        v25 = v24;
        do
        {
          v26 = v25;
          v25 = (_QWORD *)*(v25 - 6);
          sub_15F20C0(v26, v19, v11, v18);
        }
        while ( v30 != v25 );
        v9 = v33;
        v30 = 0;
        v11 = *(unsigned int *)(v33 + 8);
      }
      goto LABEL_9;
    }
    if ( a2 )
      return (__int64)a2;
  }
  v11 = *(unsigned int *)(a4 + 8);
LABEL_9:
  v20 = sub_14AC030(v7, *(unsigned int **)v9, v11, 0);
  if ( !v20 )
    return 0;
  v21 = *(unsigned int *)(v9 + 8);
  v36 = 1;
  v35 = 3;
  v34 = "tmp";
  v22 = v21 - v8;
  v23 = *(_QWORD *)v9 + 4LL * v8;
  result = sub_1648A60(88, 2);
  if ( result )
  {
    v32 = result;
    sub_15F1EA0(result, *v30, 63, result - 48, 2, v6);
    *(_QWORD *)(v32 + 64) = 0x400000000LL;
    *(_QWORD *)(v32 + 56) = v32 + 72;
    sub_15FAD90(v32, v30, v20, v23, v22, &v34);
    return v32;
  }
  return result;
}
