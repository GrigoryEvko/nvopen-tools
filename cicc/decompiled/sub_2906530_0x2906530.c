// Function: sub_2906530
// Address: 0x2906530
//
__int64 __fastcall sub_2906530(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // rcx
  __int64 *v9; // rdx
  __int64 v10; // r10
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // rax
  int v21; // edx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // edx
  int v28; // r10d
  int v29; // r11d
  __int64 v30[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = a3;
  v5 = *(unsigned int *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 8);
  v30[0] = a1;
  if ( (_DWORD)v5 )
  {
    v7 = (unsigned int)(v5 - 1);
    v8 = (unsigned int)v7 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v9 = (__int64 *)(v6 + 16 * v8);
    v10 = *v9;
    if ( a1 == *v9 )
    {
LABEL_3:
      if ( v9 != (__int64 *)(v6 + 16 * v5) )
        goto LABEL_4;
    }
    else
    {
      v21 = 1;
      while ( v10 != -4096 )
      {
        v29 = v21 + 1;
        v8 = (unsigned int)v7 & (v21 + (_DWORD)v8);
        v9 = (__int64 *)(v6 + 16LL * (unsigned int)v8);
        v10 = *v9;
        if ( a1 == *v9 )
          goto LABEL_3;
        v21 = v29;
      }
    }
  }
  v22 = sub_2906080(a1, a2, v3);
  *(_QWORD *)sub_1152A40(a2, v30, v23, v24, v25, v26) = v22;
LABEL_4:
  v11 = (__int64 *)sub_1152A40(a2, v30, (__int64)v9, v8, v3, v7);
  v12 = *(_QWORD *)(a2 + 8);
  v13 = *v11;
  v14 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v14 )
  {
    v15 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v16 = (__int64 *)(v12 + 16LL * v15);
    v17 = *v16;
    if ( v13 == *v16 )
    {
LABEL_6:
      if ( v16 != (__int64 *)(v12 + 16 * v14) )
      {
        v18 = *(_QWORD *)(a2 + 32);
        v19 = v18 + 16LL * *((unsigned int *)v16 + 2);
        if ( v19 != 16LL * *(unsigned int *)(a2 + 40) + v18 )
          return *(_QWORD *)(v19 + 8);
      }
    }
    else
    {
      v27 = 1;
      while ( v17 != -4096 )
      {
        v28 = v27 + 1;
        v15 = (v14 - 1) & (v27 + v15);
        v16 = (__int64 *)(v12 + 16LL * v15);
        v17 = *v16;
        if ( v13 == *v16 )
          goto LABEL_6;
        v27 = v28;
      }
    }
  }
  return v13;
}
