// Function: sub_22B7360
// Address: 0x22b7360
//
__int64 __fastcall sub_22B7360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 **a11,
        __int64 a12)
{
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r12
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r11
  int v18; // r15d
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 v25; // rcx
  int v26; // r8d
  __int64 v27; // rsi
  __int64 v28; // rdi
  int v29; // eax
  int v30; // eax
  int v31; // r10d
  int v33; // r9d
  __int64 *v34; // [rsp+8h] [rbp-48h]
  int v35; // [rsp+1Ch] [rbp-34h]

  v12 = *(__int64 **)a8;
  v13 = *(_QWORD *)(a8 + 8);
  v14 = *a11;
  if ( (_DWORD)v13 )
  {
    v34 = &v12[(unsigned int)v13];
    do
    {
      v19 = *(unsigned int *)(a7 + 48);
      v20 = *v12;
      v21 = *(_QWORD *)(a7 + 32);
      if ( (_DWORD)v19 )
      {
        v22 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( v20 == *v23 )
          goto LABEL_9;
        v29 = 1;
        while ( v24 != -4096 )
        {
          v33 = v29 + 1;
          v22 = (v19 - 1) & (v29 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v20 == *v23 )
            goto LABEL_9;
          v29 = v33;
        }
      }
      v23 = (__int64 *)(v21 + 16 * v19);
LABEL_9:
      v25 = *(unsigned int *)(a10 + 48);
      v26 = *((_DWORD *)v23 + 2);
      v27 = *v14;
      v28 = *(_QWORD *)(a10 + 32);
      if ( !(_DWORD)v25 )
        goto LABEL_10;
      v15 = (v25 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v16 = (__int64 *)(v28 + 16LL * v15);
      v17 = *v16;
      if ( v27 != *v16 )
      {
        v30 = 1;
        while ( v17 != -4096 )
        {
          v31 = v30 + 1;
          v15 = (v25 - 1) & (v30 + v15);
          v16 = (__int64 *)(v28 + 16LL * v15);
          v17 = *v16;
          if ( v27 == *v16 )
            goto LABEL_4;
          v30 = v31;
        }
LABEL_10:
        v16 = (__int64 *)(v28 + 16 * v25);
      }
LABEL_4:
      v18 = *((_DWORD *)v16 + 2);
      v35 = v26;
      if ( !(unsigned __int8)sub_22B6E40(a9, v26, v18) || !(unsigned __int8)sub_22B6E40(a12, v18, v35) )
        return 0;
      ++v12;
      ++v14;
    }
    while ( v12 != v34 );
  }
  return 1;
}
