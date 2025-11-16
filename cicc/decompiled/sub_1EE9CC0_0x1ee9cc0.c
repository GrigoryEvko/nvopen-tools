// Function: sub_1EE9CC0
// Address: 0x1ee9cc0
//
__int64 __fastcall sub_1EE9CC0(__int64 a1, __int64 a2, __int64 a3, int a4, unsigned int a5, int a6)
{
  __int64 v8; // r13
  __int64 (*v9)(void); // rdx
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 (*v12)(void); // rax
  __int64 v13; // rax
  __int64 v14; // r13
  unsigned int v15; // r12d
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 result; // rax
  __int64 i; // rdx
  unsigned __int64 v20; // rax
  unsigned int v21; // r13d
  int v22; // r12d
  unsigned int v23; // esi
  int v24; // ecx
  int v25; // r8d
  int v26; // r9d
  int v27; // ecx
  int v28; // r8d
  int v29; // r9d
  unsigned __int64 v30; // rdx
  unsigned int v31; // r13d
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // r15
  __int64 v34; // rax
  int v35; // ecx
  __int64 v36; // rdx
  unsigned __int64 v37; // r15
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned int v40; // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 56);
  v9 = *(__int64 (**)(void))(**(_QWORD **)(v8 + 16) + 40LL);
  v10 = 0;
  if ( v9 != sub_1D00B00 )
    v10 = v9();
  *(_QWORD *)(a1 + 8) = v10;
  v11 = 0;
  v12 = *(__int64 (**)(void))(**(_QWORD **)(v8 + 16) + 112LL);
  if ( v12 != sub_1D00B10 )
    v11 = v12();
  *(_QWORD *)a1 = v11;
  v13 = *(_QWORD *)(v8 + 40);
  v14 = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 96) = v11;
  *(_QWORD *)(a1 + 16) = v13;
  if ( v14 )
  {
    memset(*(void **)(a1 + 104), 0, 8 * v14);
    v14 = *(_QWORD *)(a1 + 112);
  }
  v15 = *(_DWORD *)(v11 + 44);
  if ( v15 > (unsigned __int64)(v14 << 6) )
  {
    v32 = *(_QWORD *)(a1 + 104);
    v33 = (v15 + 63) >> 6;
    if ( v33 < 2 * v14 )
      v33 = 2 * v14;
    v34 = (__int64)realloc(v32, 8 * v33, 8 * (int)v33, a4, a5, a6);
    if ( !v34 )
    {
      if ( 8 * v33 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v34 = 0;
      }
      else
      {
        v34 = sub_13A3880(1u);
      }
    }
    v35 = *(_DWORD *)(a1 + 120);
    *(_QWORD *)(a1 + 104) = v34;
    *(_QWORD *)(a1 + 112) = v33;
    a5 = (unsigned int)(v35 + 63) >> 6;
    if ( v33 > a5 )
    {
      v37 = v33 - a5;
      if ( v37 )
      {
        v40 = (unsigned int)(v35 + 63) >> 6;
        memset((void *)(v34 + 8LL * a5), 0, 8 * v37);
        v35 = *(_DWORD *)(a1 + 120);
        v34 = *(_QWORD *)(a1 + 104);
        a5 = v40;
      }
    }
    a4 = v35 & 0x3F;
    if ( a4 )
    {
      *(_QWORD *)(v34 + 8LL * (a5 - 1)) &= ~(-1LL << a4);
      v34 = *(_QWORD *)(a1 + 104);
    }
    v36 = *(_QWORD *)(a1 + 112) - (unsigned int)v14;
    if ( v36 )
      memset((void *)(v34 + 8LL * (unsigned int)v14), 0, 8 * v36);
  }
  v16 = *(_DWORD *)(a1 + 120);
  if ( v15 > v16 )
  {
    v30 = *(_QWORD *)(a1 + 112);
    v31 = (v16 + 63) >> 6;
    if ( v30 > v31 )
    {
      v38 = v30 - v31;
      if ( v38 )
      {
        memset((void *)(*(_QWORD *)(a1 + 104) + 8LL * v31), 0, 8 * v38);
        v16 = *(_DWORD *)(a1 + 120);
      }
    }
    a4 = v16 & 0x3F;
    if ( (v16 & 0x3F) != 0 )
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (v31 - 1)) &= ~(-1LL << a4);
      v16 = *(_DWORD *)(a1 + 120);
    }
  }
  *(_DWORD *)(a1 + 120) = v15;
  if ( v16 <= v15 )
    goto LABEL_40;
  v20 = *(_QWORD *)(a1 + 112);
  v21 = (v15 + 63) >> 6;
  a4 = v21;
  if ( v20 > v21 )
  {
    v39 = v20 - v21;
    if ( v39 )
    {
      memset((void *)(*(_QWORD *)(a1 + 104) + 8LL * v21), 0, 8 * v39);
      v15 = *(_DWORD *)(a1 + 120);
    }
  }
  v22 = v15 & 0x3F;
  if ( v22 )
  {
    a4 = v22;
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * (v21 - 1)) &= ~(-1LL << v22);
    if ( *(_QWORD *)(a1 + 24) )
      goto LABEL_11;
  }
  else
  {
LABEL_40:
    if ( *(_QWORD *)(a1 + 24) )
      goto LABEL_11;
  }
  v23 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
  *(_DWORD *)(a1 + 40) = v23;
  sub_13A49F0(a1 + 128, v23, 0, a4, a5, a6);
  sub_13A49F0(a1 + 152, *(_DWORD *)(a1 + 40), 0, v24, v25, v26);
  sub_13A49F0(a1 + 176, *(_DWORD *)(a1 + 40), 0, v27, v28, v29);
LABEL_11:
  v17 = *(unsigned int *)(a1 + 56);
  result = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 24) = a2;
  for ( i = result + 16 * v17; i != result; *(_QWORD *)(result - 8) = 0 )
  {
    *(_DWORD *)(result + 4) = 0;
    result += 16;
  }
  *(_BYTE *)(a1 + 44) = 0;
  return result;
}
