// Function: sub_14C9920
// Address: 0x14c9920
//
__int64 __fastcall sub_14C9920(
        __int64 a1,
        _DWORD *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(__int64, __int64, __int64, __int64, __int64),
        __int64 a6)
{
  __int64 v10; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // r14
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 v29; // rax
  int v30; // edx
  int v31; // r9d
  _DWORD *v32; // [rsp+8h] [rbp-58h]
  _DWORD *v33; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF
  __int64 v37; // [rsp+28h] [rbp-38h]

  if ( a4 )
  {
    if ( *(_BYTE *)(a4 + 184) )
    {
      v10 = *(unsigned int *)(a4 + 176);
      if ( !(_DWORD)v10 )
        return 0;
    }
    else
    {
      sub_14CDF70(a4);
      v10 = *(unsigned int *)(a4 + 176);
      if ( !(_DWORD)v10 )
        return 0;
    }
    v12 = *(_QWORD *)(a4 + 160);
    v13 = (v10 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v14 = v12 + 88LL * v13;
    v15 = *(_QWORD *)(v14 + 24);
    if ( a1 != v15 )
    {
      v30 = 1;
      while ( v15 != -8 )
      {
        v31 = v30 + 1;
        v13 = (v10 - 1) & (v30 + v13);
        v14 = v12 + 88LL * v13;
        v15 = *(_QWORD *)(v14 + 24);
        if ( a1 == v15 )
          goto LABEL_6;
        v30 = v31;
      }
      return 0;
    }
LABEL_6:
    if ( v14 == v12 + 88 * v10 )
      return 0;
    v16 = *(_QWORD *)(v14 + 40);
    v17 = v16 + 32LL * *(unsigned int *)(v14 + 48);
    if ( v17 == v16 )
      return 0;
    v32 = &a2[a3];
    while ( 1 )
    {
      v18 = *(_QWORD *)(v16 + 16);
      if ( v18 )
      {
        v19 = *(unsigned int *)(v16 + 24);
        if ( (_DWORD)v19 != -1 )
        {
          v20 = 0;
          if ( *(char *)(v18 + 23) < 0 )
          {
            v20 = sub_1648A40(*(_QWORD *)(v16 + 16));
            v19 = *(unsigned int *)(v16 + 24);
          }
          v36 = sub_14C9840(v18, v20 + 16 * v19);
          v37 = v21;
          if ( (_DWORD)v36 )
          {
            if ( v21 == a1 && v32 != sub_14C8440(a2, (__int64)v32, (int *)&v36) )
            {
              v22 = 0;
              if ( *(char *)(v18 + 23) < 0 )
                v22 = sub_1648A40(v18);
              if ( a5(a6, v36, v37, v18, v22 + 16LL * *(unsigned int *)(v16 + 24)) )
                break;
            }
          }
        }
      }
      v16 += 32;
      if ( v17 == v16 )
        return 0;
    }
  }
  else
  {
    v23 = *(_QWORD **)(a1 + 8);
    if ( !v23 )
      return 0;
    v33 = &a2[a3];
    while ( 1 )
    {
      v24 = sub_14C8500(v23);
      if ( v24 )
      {
        v25 = sub_1648700(v23);
        v36 = sub_14C9840(v25, v24);
        v26 = v36;
        v28 = v27;
        v37 = v27;
        if ( (_DWORD)v36 )
        {
          if ( v33 != sub_14C8440(a2, (__int64)v33, (int *)&v36) )
          {
            v29 = sub_1648700(v23);
            if ( a5(a6, v26, v28, v29, v24) )
              break;
          }
        }
      }
      v23 = (_QWORD *)v23[1];
      if ( !v23 )
        return 0;
    }
  }
  return v36;
}
