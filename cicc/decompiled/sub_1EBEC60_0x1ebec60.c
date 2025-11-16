// Function: sub_1EBEC60
// Address: 0x1ebec60
//
__int64 __fastcall sub_1EBEC60(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // eax
  __int64 v7; // r15
  __int64 v8; // r8
  __int64 v11; // r13
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r12
  unsigned int v16; // r12d
  __int64 v17; // rcx
  __int64 v18; // rax
  _QWORD *v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rdx
  int v22; // eax
  __int64 v23; // rsi
  int v24; // ecx
  unsigned int v25; // eax
  __int64 *v26; // rdx
  __int64 v27; // rdi
  unsigned int v28; // eax
  _QWORD *v29; // rdi
  __int64 v30; // rsi
  _QWORD *v31; // rax
  int v32; // r8d
  int v33; // edx
  int v34; // r8d
  __int64 v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a2 & 0x7FFFFFFF;
  v7 = a2 & 0x7FFFFFFF;
  v8 = 8 * v7;
  v11 = *(_QWORD *)(a1 + 264);
  v12 = *(unsigned int *)(v11 + 408);
  if ( (a2 & 0x7FFFFFFFu) >= (unsigned int)v12
    || (v13 = *(_QWORD *)(v11 + 400), (v14 = *(_QWORD *)(v13 + 8LL * v6)) == 0) )
  {
    v16 = v6 + 1;
    if ( (unsigned int)v12 < v6 + 1 )
    {
      v18 = v16;
      if ( v16 < v12 )
      {
        *(_DWORD *)(v11 + 408) = v16;
      }
      else if ( v16 > v12 )
      {
        if ( v16 > (unsigned __int64)*(unsigned int *)(v11 + 412) )
        {
          sub_16CD150(v11 + 400, (const void *)(v11 + 416), v16, 8, 8 * a2, a6);
          v12 = *(unsigned int *)(v11 + 408);
          v8 = 8LL * (a2 & 0x7FFFFFFF);
          v18 = v16;
        }
        v17 = *(_QWORD *)(v11 + 400);
        v19 = (_QWORD *)(v17 + 8 * v18);
        v20 = (_QWORD *)(v17 + 8 * v12);
        v21 = *(_QWORD *)(v11 + 416);
        if ( v19 != v20 )
        {
          do
            *v20++ = v21;
          while ( v19 != v20 );
          v17 = *(_QWORD *)(v11 + 400);
        }
        *(_DWORD *)(v11 + 408) = v16;
        goto LABEL_7;
      }
    }
    v17 = *(_QWORD *)(v11 + 400);
LABEL_7:
    *(_QWORD *)(v17 + v8) = sub_1DBA290(a2);
    v14 = *(_QWORD *)(*(_QWORD *)(v11 + 400) + 8 * v7);
    sub_1DBB110((_QWORD *)v11, v14);
  }
  if ( !*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 264LL) + 4 * v7) )
  {
    *(_DWORD *)(v14 + 72) = 0;
    *(_DWORD *)(v14 + 8) = 0;
    return 0;
  }
  sub_21031A0(*(_QWORD *)(a1 + 272), v14, v12, v13, v8);
  v35[0] = v14;
  if ( (*(_BYTE *)(a1 + 27424) & 1) != 0 )
  {
    v23 = a1 + 27432;
    v24 = 7;
LABEL_19:
    v25 = v24 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v26 = (__int64 *)(v23 + 8LL * v25);
    v27 = *v26;
    if ( v14 == *v26 )
    {
LABEL_20:
      *v26 = -16;
      v28 = *(_DWORD *)(a1 + 27424);
      ++*(_DWORD *)(a1 + 27428);
      v29 = *(_QWORD **)(a1 + 27496);
      *(_DWORD *)(a1 + 27424) = (2 * (v28 >> 1) - 2) | v28 & 1;
      v30 = (__int64)&v29[*(unsigned int *)(a1 + 27504)];
      v31 = sub_1EBB320(v29, v30, v35);
      if ( v31 + 1 != (_QWORD *)v30 )
      {
        memmove(v31, v31 + 1, v30 - (_QWORD)(v31 + 1));
        v32 = *(_DWORD *)(a1 + 27504);
      }
      *(_DWORD *)(a1 + 27504) = v32 - 1;
    }
    else
    {
      v33 = 1;
      while ( v27 != -8 )
      {
        v34 = v33 + 1;
        v25 = v24 & (v33 + v25);
        v26 = (__int64 *)(v23 + 8LL * v25);
        v27 = *v26;
        if ( v14 == *v26 )
          goto LABEL_20;
        v33 = v34;
      }
    }
    return 1;
  }
  v22 = *(_DWORD *)(a1 + 27440);
  v23 = *(_QWORD *)(a1 + 27432);
  if ( v22 )
  {
    v24 = v22 - 1;
    goto LABEL_19;
  }
  return 1;
}
