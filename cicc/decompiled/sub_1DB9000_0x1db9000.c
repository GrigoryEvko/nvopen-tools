// Function: sub_1DB9000
// Address: 0x1db9000
//
void __fastcall sub_1DB9000(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int *v7; // rsi
  unsigned int v10; // r15d
  __int64 v11; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 i; // rdx
  __int64 v18; // rdi
  unsigned int *v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // r10
  _DWORD *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // r14
  unsigned __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // rdx
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // r10
  __int64 v32; // rax
  __int64 v33; // r11
  _QWORD *v34; // rcx
  __int64 v35; // rcx
  __int64 v36; // [rsp-10h] [rbp-210h]
  __int64 v37; // [rsp+0h] [rbp-200h]
  unsigned int v38; // [rsp+8h] [rbp-1F8h]
  int v39; // [rsp+Ch] [rbp-1F4h]
  _DWORD *v40; // [rsp+10h] [rbp-1F0h]
  unsigned int v41; // [rsp+18h] [rbp-1E8h]
  _QWORD v42[4]; // [rsp+20h] [rbp-1E0h] BYREF
  _BYTE *v43; // [rsp+40h] [rbp-1C0h]
  __int64 v44; // [rsp+48h] [rbp-1B8h]
  _BYTE v45[432]; // [rsp+50h] [rbp-1B0h] BYREF

  v7 = (int *)a4;
  v10 = *(_DWORD *)(a1 + 72);
  v11 = *(unsigned int *)(a5 + 8);
  if ( v10 )
  {
    a6 = v10;
    v13 = 0;
    while ( 1 )
    {
      v15 = *(unsigned int *)(a3 + 4 * v13);
      if ( (_DWORD)v15 != (_DWORD)v13 )
        break;
      a4 = *(_QWORD *)a5;
      v14 = *(_QWORD *)(*(_QWORD *)a5 + 8 * v15);
      if ( v14 )
      {
        a4 = *(_QWORD *)(a1 + 64);
        if ( v14 != *(_QWORD *)(a4 + 8 * v13) )
          break;
      }
      if ( v10 == ++v13 )
        goto LABEL_8;
    }
    if ( *(_DWORD *)(a1 + 8) )
    {
      v30 = *(_QWORD **)a1;
      v31 = *(_QWORD *)(*(_QWORD *)a5 + 8LL * *(int *)(a3 + 4LL * **(unsigned int **)(*(_QWORD *)a1 + 16LL)));
      v32 = *(_QWORD *)a1 + 24LL;
      *(_QWORD *)(*(_QWORD *)a1 + 16LL) = v31;
      a4 = *(_QWORD *)a1;
      a6 = 3LL * *(unsigned int *)(a1 + 8);
      v33 = *(_QWORD *)a1 + 24LL * *(unsigned int *)(a1 + 8);
      if ( v33 != v32 )
      {
        while ( 1 )
        {
          a6 = *(_QWORD *)(*(_QWORD *)a5 + 8LL * *(int *)(a3 + 4LL * **(unsigned int **)(v32 + 16)));
          if ( a6 == v31 && v30[1] == *(_QWORD *)v32 )
          {
            v35 = *(_QWORD *)(v32 + 8);
            v32 += 24;
            v30[1] = v35;
            v34 = v30;
            if ( v32 == v33 )
              goto LABEL_44;
          }
          else
          {
            v34 = v30 + 3;
            v30[5] = a6;
            if ( v30 + 3 != (_QWORD *)v32 )
            {
              v30[3] = *(_QWORD *)v32;
              a6 = *(_QWORD *)(v32 + 8);
              v30[4] = a6;
            }
            v32 += 24;
            if ( v32 == v33 )
            {
LABEL_44:
              v32 = (__int64)(v34 + 3);
              a4 = *(_QWORD *)a1;
              break;
            }
          }
          v31 = v34[2];
          v30 = v34;
        }
      }
      *(_DWORD *)(a1 + 8) = -1431655765 * ((v32 - a4) >> 3);
    }
  }
LABEL_8:
  v16 = *a2;
  i = 3LL * *((unsigned int *)a2 + 2);
  v18 = *a2 + 24LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v18 )
  {
    do
    {
      v19 = *(unsigned int **)(v16 + 16);
      v16 += 24;
      a4 = v7[*v19];
      i = *(_QWORD *)(*(_QWORD *)a5 + 8 * a4);
      *(_QWORD *)(v16 - 8) = i;
    }
    while ( v18 != v16 );
  }
  if ( (_DWORD)v11 )
  {
    v20 = 0;
    v7 = (int *)(a1 + 80);
    a6 = 0;
    v21 = 8LL * (unsigned int)v11;
    do
    {
      v22 = *(_DWORD **)(*(_QWORD *)a5 + v20);
      if ( v22 )
      {
        if ( v10 > (unsigned int)a6 )
        {
          a4 = *(_QWORD *)(a1 + 64);
          i = (unsigned int)a6;
          *(_QWORD *)(a4 + 8LL * (unsigned int)a6) = v22;
        }
        else
        {
          i = *(unsigned int *)(a1 + 72);
          if ( (unsigned int)i >= *(_DWORD *)(a1 + 76) )
          {
            v37 = v21;
            v38 = v11;
            v39 = a6;
            v40 = *(_DWORD **)(*(_QWORD *)a5 + v20);
            sub_16CD150(a1 + 64, v7, 0, 8, v11, a6);
            i = *(unsigned int *)(a1 + 72);
            v21 = v37;
            v11 = v38;
            LODWORD(a6) = v39;
            v22 = v40;
          }
          a4 = *(_QWORD *)(a1 + 64);
          *(_QWORD *)(a4 + 8 * i) = v22;
          ++*(_DWORD *)(a1 + 72);
        }
        *v22 = a6;
        a6 = (unsigned int)(a6 + 1);
      }
      v20 += 8;
    }
    while ( v21 != v20 );
  }
  if ( (unsigned int)v11 < v10 )
  {
    v26 = *(unsigned int *)(a1 + 72);
    v27 = (unsigned int)v11;
    if ( (unsigned int)v11 >= v26 )
    {
      if ( (unsigned int)v11 <= v26 )
        goto LABEL_21;
      if ( (unsigned int)v11 > (unsigned __int64)*(unsigned int *)(a1 + 76) )
      {
        v7 = (int *)(a1 + 80);
        v41 = v11;
        sub_16CD150(a1 + 64, (const void *)(a1 + 80), (unsigned int)v11, 8, v11, a6);
        v26 = *(unsigned int *)(a1 + 72);
        v11 = v41;
      }
      v28 = *(_QWORD *)(a1 + 64);
      v29 = (_QWORD *)(v28 + 8 * v26);
      for ( i = v28 + 8 * v27; (_QWORD *)i != v29; ++v29 )
      {
        if ( v29 )
          *v29 = 0;
      }
    }
    *(_DWORD *)(a1 + 72) = v11;
  }
LABEL_21:
  v23 = *a2;
  v42[0] = a1;
  v44 = 0x1000000000LL;
  v24 = *((unsigned int *)a2 + 2);
  v42[1] = 0;
  v43 = v45;
  v25 = v23 + 24 * v24;
  while ( v25 != v23 )
  {
    v36 = *(_QWORD *)(v23 + 16);
    v23 += 24;
    sub_1DB8AC0((__int64)v42, (__int64)v7, i, a4, v11, (_QWORD *)a6, *(_OWORD *)(v23 - 24), v36);
  }
  sub_1DB55F0((__int64)v42);
  if ( v43 != v45 )
    _libc_free((unsigned __int64)v43);
}
