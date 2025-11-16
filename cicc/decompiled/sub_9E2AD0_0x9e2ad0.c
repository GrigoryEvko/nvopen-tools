// Function: sub_9E2AD0
// Address: 0x9e2ad0
//
__int64 __fastcall sub_9E2AD0(__int64 a1, __int64 *a2, __int64 a3, _DWORD *a4)
{
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rdx
  int v19; // r14d
  __int64 v20; // rax
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rdx
  int v24; // [rsp+8h] [rbp-48h]
  int v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  int v27; // [rsp+18h] [rbp-38h]
  int v28; // [rsp+18h] [rbp-38h]
  int v29; // [rsp+18h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 16;
  v26 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0xC00000000LL;
  v7 = a2[78] == a2[77];
  v8 = (unsigned int)*a4;
  *a4 = v8 + 1;
  v9 = *(_QWORD *)(a3 + 8 * v8);
  if ( v7 )
  {
    v29 = v9;
    if ( *(_DWORD *)(a1 + 12) < (unsigned int)v9 )
    {
      v25 = v9;
      sub_C8D5F0(a1, v26, (unsigned int)v9, 4);
      LODWORD(v9) = v25;
    }
    v19 = 0;
    if ( (_DWORD)v9 )
    {
      do
      {
        v20 = (unsigned int)*a4;
        v21 = a2[53];
        *a4 = v20 + 1;
        v22 = sub_9E27D0(v21, *(_QWORD *)(a2[74] + 8LL * *(_QWORD *)(a3 + 8 * v20)));
        v23 = *(unsigned int *)(a1 + 8);
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v24 = v22;
          sub_C8D5F0(a1, v26, v23 + 1, 4);
          v23 = *(unsigned int *)(a1 + 8);
          v22 = v24;
        }
        ++v19;
        *(_DWORD *)(*(_QWORD *)a1 + 4 * v23) = v22;
        ++*(_DWORD *)(a1 + 8);
      }
      while ( v29 != v19 );
    }
  }
  else
  {
    v10 = v9 + 1;
    v11 = *(_QWORD *)(a2[77] + 8LL * (unsigned int)v9);
    v12 = v11;
    if ( *(_DWORD *)(a1 + 12) < (unsigned int)v11 )
    {
      v28 = v11;
      sub_C8D5F0(a1, v26, (unsigned int)v11, 4);
      LODWORD(v11) = v28;
    }
    if ( (_DWORD)v11 )
    {
      do
      {
        v13 = a2[77];
        v14 = *(_QWORD *)(v13 + 8LL * v10);
        v15 = (unsigned int)v14;
        if ( (int)v14 < 0 )
        {
          v10 -= v14;
          v15 = *(unsigned int *)(v13 + 8LL * v10);
        }
        ++v10;
        v16 = sub_9E27D0(a2[53], *(_QWORD *)(a2[74] + 8 * v15));
        v17 = *(unsigned int *)(a1 + 8);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v27 = v16;
          sub_C8D5F0(a1, v26, v17 + 1, 4);
          v17 = *(unsigned int *)(a1 + 8);
          v16 = v27;
        }
        *(_DWORD *)(*(_QWORD *)a1 + 4 * v17) = v16;
        ++*(_DWORD *)(a1 + 8);
        --v12;
      }
      while ( v12 );
    }
  }
  return a1;
}
