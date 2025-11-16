// Function: sub_1DB4D80
// Address: 0x1db4d80
//
unsigned __int64 __fastcall sub_1DB4D80(__int64 a1, __int64 a2, int a3, _QWORD *a4, __int64 a5)
{
  __int64 v5; // r15
  int v10; // eax
  __int64 v11; // r9
  int v12; // edx
  __int64 (*v13)(void); // rax
  unsigned __int64 result; // rax
  __int64 v15; // rbx
  int v16; // r13d
  char v17; // dl
  char v18; // cl
  unsigned __int64 i; // rdx
  __int64 v20; // rsi
  __int64 v21; // r8
  unsigned int v22; // edi
  __int64 *v23; // rax
  __int64 v24; // r11
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // rax
  int v28; // r10d
  int v29; // [rsp+8h] [rbp-38h]
  unsigned __int64 v30; // [rsp+8h] [rbp-38h]

  v5 = 0;
  v10 = sub_1E69F40(a4, *(unsigned int *)(a1 + 112));
  v11 = a1;
  v12 = v10;
  v13 = *(__int64 (**)(void))(**(_QWORD **)(*a4 + 16LL) + 112LL);
  if ( v13 != sub_1D00B10 )
  {
    v29 = v12;
    v27 = v13();
    v11 = a1;
    v12 = v29;
    v5 = v27;
  }
  result = *(unsigned int *)(v11 + 112);
  if ( (result & 0x80000000) != 0LL )
  {
    result = a4[3] + 16 * (result & 0x7FFFFFFF);
    v15 = *(_QWORD *)(result + 8);
  }
  else
  {
    v15 = *(_QWORD *)(a4[34] + 8 * result);
  }
  if ( v15 )
  {
    if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 || (v15 = *(_QWORD *)(v15 + 32)) != 0 && (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
    {
      v16 = v12 & a3;
      while ( 1 )
      {
        v17 = *(_BYTE *)(v15 + 4);
        if ( (v17 & 1) != 0 )
        {
          result = (unsigned int)~*(_DWORD *)(*(_QWORD *)(v5 + 248) + 4LL * ((*(_DWORD *)v15 >> 8) & 0xFFF));
          if ( (v16 & (unsigned int)result) != 0 )
            break;
        }
LABEL_10:
        v15 = *(_QWORD *)(v15 + 32);
        if ( !v15 || (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
          return result;
      }
      v18 = v17 & 4;
      for ( i = *(_QWORD *)(v15 + 16); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v20 = *(unsigned int *)(a5 + 384);
      v21 = *(_QWORD *)(a5 + 368);
      if ( (_DWORD)v20 )
      {
        LODWORD(v11) = v20 - 1;
        v22 = (v20 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( i == *v23 )
        {
LABEL_17:
          result = (v18 == 0 ? 4LL : 2LL) | v23[1] & 0xFFFFFFFFFFFFFFF8LL;
          v25 = *(unsigned int *)(a2 + 8);
          if ( (unsigned int)v25 >= *(_DWORD *)(a2 + 12) )
          {
            v30 = result;
            sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v21, v11);
            v25 = *(unsigned int *)(a2 + 8);
            result = v30;
          }
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v25) = result;
          ++*(_DWORD *)(a2 + 8);
          goto LABEL_10;
        }
        v26 = 1;
        while ( v24 != -8 )
        {
          v28 = v26 + 1;
          v22 = v11 & (v26 + v22);
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( i == *v23 )
            goto LABEL_17;
          v26 = v28;
        }
      }
      v23 = (__int64 *)(v21 + 16 * v20);
      goto LABEL_17;
    }
  }
  return result;
}
