// Function: sub_2A117C0
// Address: 0x2a117c0
//
__int64 __fastcall sub_2A117C0(
        __int64 a1,
        int a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64),
        __int64 a8)
{
  __int64 v10; // r15
  __int64 v12; // rax
  _QWORD *v13; // r14
  int v14; // ecx
  __int64 v15; // rsi
  int v16; // ecx
  unsigned int v17; // edx
  __int64 *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rbx
  _QWORD *v21; // rdi
  __int64 v22; // rax
  int v23; // ecx
  __int64 v24; // rsi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r8
  __int64 v29; // rsi
  __int64 v30; // rax
  int v32; // eax
  int v33; // r9d
  int v34; // eax
  int v35; // r8d

  if ( !a1 || *(_QWORD *)(a1 + 8) != *(_QWORD *)(a3 + 8) )
    return 0;
  v10 = a1;
  if ( a4 != a2 )
  {
    v12 = a7(a8);
    v13 = (_QWORD *)v12;
    if ( !v12 )
      return 0;
    v14 = *(_DWORD *)(v12 + 56);
    v15 = *(_QWORD *)(v12 + 40);
    if ( v14 )
    {
      v16 = v14 - 1;
      v17 = v16 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v18 = (__int64 *)(v15 + 16LL * v17);
      v19 = *v18;
      if ( a1 == *v18 )
      {
LABEL_7:
        v20 = v18[1];
        goto LABEL_8;
      }
      v34 = 1;
      while ( v19 != -4096 )
      {
        v35 = v34 + 1;
        v17 = v16 & (v34 + v17);
        v18 = (__int64 *)(v15 + 16LL * v17);
        v19 = *v18;
        if ( a1 == *v18 )
          goto LABEL_7;
        v34 = v35;
      }
    }
    v20 = 0;
LABEL_8:
    v21 = sub_103E0E0(v13);
    v22 = v21[1];
    v23 = *(_DWORD *)(v22 + 56);
    v24 = *(_QWORD *)(v22 + 40);
    if ( v23 )
    {
      v25 = v23 - 1;
      v26 = v25 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v27 = (__int64 *)(v24 + 16LL * v26);
      v28 = *v27;
      if ( a3 == *v27 )
      {
LABEL_10:
        v29 = v27[1];
        goto LABEL_11;
      }
      v32 = 1;
      while ( v28 != -4096 )
      {
        v33 = v32 + 1;
        v26 = v25 & (v32 + v26);
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( a3 == *v27 )
          goto LABEL_10;
        v32 = v33;
      }
    }
    v29 = 0;
LABEL_11:
    v30 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v21 + 16LL))(v21, v29, a5);
    if ( sub_1041420((__int64)v13, v30, v20) )
      return v10;
    return 0;
  }
  return v10;
}
