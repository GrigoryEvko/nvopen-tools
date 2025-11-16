// Function: sub_1D95C60
// Address: 0x1d95c60
//
__int64 __fastcall sub_1D95C60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, int a6)
{
  _QWORD *v7; // rax
  __int64 v8; // r13
  __int16 v9; // ax
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // rcx
  int v15; // r15d
  unsigned int v16; // eax
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  const void *v19; // r8
  size_t v20; // r14
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  const void *v24; // r14
  size_t v25; // r15
  unsigned __int64 v26; // r12
  int v27; // eax
  unsigned __int8 v28; // dl
  __int64 result; // rax
  _BYTE *v30; // r14
  _BYTE *v31; // r13
  size_t v32; // r15
  char *v33; // r12
  __int64 v34; // r13
  char *i; // r14
  _QWORD *v37; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  __int64 v41; // [rsp+28h] [rbp-48h]
  _QWORD *v42; // [rsp+30h] [rbp-40h]
  char *desta; // [rsp+38h] [rbp-38h]
  void *destb; // [rsp+38h] [rbp-38h]

  v41 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 56LL);
  v7 = *(_QWORD **)(a3 + 16);
  v40 = a1 + 256;
  v37 = v7;
  v42 = v7 + 3;
  v8 = v7[4];
  if ( v7 + 3 == (_QWORD *)v8 )
  {
LABEL_17:
    if ( !a5 )
    {
      v30 = (_BYTE *)v37[12];
      v31 = (_BYTE *)v37[11];
      v32 = v30 - v31;
      if ( (unsigned __int64)(v30 - v31) > 0x7FFFFFFFFFFFFFF8LL )
        sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
      desta = 0;
      if ( v32 )
        desta = (char *)sub_22077B0(v32);
      v33 = &desta[v32];
      if ( v31 != v30 )
        memcpy(desta, v31, v32);
      v34 = v37[1];
      if ( v34 == v37[7] + 320LL )
        v34 = 0;
      if ( (*(_BYTE *)a3 & 0x40) == 0 )
        v34 = 0;
      for ( i = desta; v33 != i; i += 8 )
      {
        if ( v34 != *(_QWORD *)i )
          sub_1DD8FE0(*(_QWORD *)(a2 + 16), *(_QWORD *)i, 0xFFFFFFFFLL);
      }
      if ( desta )
        j_j___libc_free_0(desta, v32);
    }
  }
  else
  {
    while ( 1 )
    {
      if ( a5 )
      {
        v9 = *(_WORD *)(v8 + 46);
        if ( (v9 & 4) != 0 || (v9 & 8) == 0 )
          v10 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v8 + 16) + 8LL) >> 7;
        else
          v10 = sub_1E15D00(v8, 128, 1);
        if ( v10 )
          break;
      }
      v11 = sub_1E0B7C0(v41, v8);
      v12 = *(_QWORD *)(a2 + 16);
      v13 = v11;
      sub_1DD5BA0(v12 + 16, v11);
      v14 = *(_QWORD *)(v12 + 24);
      *(_QWORD *)(v13 + 8) = v12 + 24;
      v14 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v13 = v14 | *(_QWORD *)v13 & 7LL;
      *(_QWORD *)(v14 + 8) = v13;
      *(_QWORD *)(v12 + 24) = v13 | *(_QWORD *)(v12 + 24) & 7LL;
      ++*(_DWORD *)(a2 + 4);
      v15 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 544) + 856LL))(*(_QWORD *)(a1 + 544), v8);
      v16 = sub_1F4BF20(v40, v8, 0);
      if ( v16 > 1 )
        *(_DWORD *)(a2 + 8) = v16 + *(_DWORD *)(a2 + 8) - 1;
      *(_DWORD *)(a2 + 12) += v15;
      v17 = *(_QWORD *)(a1 + 544);
      v18 = *(__int64 (**)())(*(_QWORD *)v17 + 656LL);
      if ( (v18 == sub_1D918C0 || !((unsigned __int8 (__fastcall *)(__int64, __int64))v18)(v17, v8))
        && (unsigned __int16)(**(_WORD **)(v13 + 16) - 12) > 1u )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 544) + 696LL))(
          *(_QWORD *)(a1 + 544),
          v13,
          *(_QWORD *)a4,
          *(unsigned int *)(a4 + 8));
      }
      sub_1D954F0(v13, a1 + 576);
      if ( !v8 )
        BUG();
      if ( (*(_BYTE *)v8 & 4) != 0 )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v42 == (_QWORD *)v8 )
          goto LABEL_17;
      }
      else
      {
        while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
        v8 = *(_QWORD *)(v8 + 8);
        if ( v42 == (_QWORD *)v8 )
          goto LABEL_17;
      }
    }
  }
  v19 = *(const void **)(a3 + 216);
  v20 = 40LL * *(unsigned int *)(a3 + 224);
  v21 = *(unsigned int *)(a3 + 224);
  v22 = *(unsigned int *)(a2 + 224);
  if ( v21 > (unsigned __int64)*(unsigned int *)(a2 + 228) - v22 )
  {
    destb = *(void **)(a3 + 216);
    sub_16CD150(a2 + 216, (const void *)(a2 + 232), v21 + v22, 40, (int)v19, a6);
    v22 = *(unsigned int *)(a2 + 224);
    v19 = destb;
  }
  if ( v20 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 216) + 40 * v22), v19, v20);
    LODWORD(v22) = *(_DWORD *)(a2 + 224);
  }
  LODWORD(v23) = v21 + v22;
  *(_DWORD *)(a2 + 224) = v23;
  v23 = (unsigned int)v23;
  v24 = *(const void **)a4;
  v25 = 40LL * *(unsigned int *)(a4 + 8);
  v26 = *(unsigned int *)(a4 + 8);
  if ( v26 > *(unsigned int *)(a2 + 228) - (unsigned __int64)(unsigned int)v23 )
  {
    sub_16CD150(a2 + 216, (const void *)(a2 + 232), (unsigned int)v23 + v26, 40, (int)v19, a6);
    v23 = *(unsigned int *)(a2 + 224);
  }
  if ( v25 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 216) + 40 * v23), v24, v25);
    LODWORD(v23) = *(_DWORD *)(a2 + 224);
  }
  *(_DWORD *)(a2 + 224) = v26 + v23;
  v27 = *(unsigned __int8 *)(a2 + 1);
  v28 = *(_BYTE *)(a3 + 1);
  *(_BYTE *)a2 &= ~4u;
  result = ((unsigned __int8)v27 | v28) & 2 | v27 & 0xFFFFFFFD;
  *(_BYTE *)(a2 + 1) = result;
  return result;
}
