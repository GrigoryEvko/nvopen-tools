// Function: sub_318E8D0
// Address: 0x318e8d0
//
__int64 __fastcall sub_318E8D0(_QWORD *a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rdi
  __int64 *v19; // r9
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  int v22; // eax
  __int64 *v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // [rsp+0h] [rbp-90h]
  __int64 v26; // [rsp+8h] [rbp-88h]
  char *v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+18h] [rbp-78h] BYREF
  __m128i v29; // [rsp+20h] [rbp-70h] BYREF
  __int64 v30; // [rsp+30h] [rbp-60h]
  __m128i v31; // [rsp+40h] [rbp-50h] BYREF
  __int64 v32; // [rsp+50h] [rbp-40h]

  v5 = a2;
  (*(void (__fastcall **)(__m128i *, _QWORD *, _QWORD, __int64))(*a1 + 16LL))(&v29, a1, a2, 1);
  v6 = a1[3];
  v7 = *(_DWORD *)(v6 + 72) == 1;
  v31 = _mm_loadu_si128(&v29);
  v32 = v30;
  if ( !v7 )
    goto LABEL_2;
  v14 = sub_22077B0(0x28u);
  v15 = v14;
  if ( v14 )
  {
    v26 = v14;
    *(__m128i *)(v14 + 8) = _mm_loadu_si128(&v31);
    *(_QWORD *)v14 = &unk_4A34750;
    *(_QWORD *)(v14 + 24) = v32;
    v16 = sub_318E5D0((__int64)&v31);
    v15 = v26;
    *(_QWORD *)(v26 + 32) = v16;
  }
  v17 = *(unsigned int *)(v6 + 16);
  v18 = *(unsigned int *)(v6 + 20);
  v28 = v15;
  v19 = &v28;
  v20 = *(_QWORD *)(v6 + 8);
  v21 = v17 + 1;
  v22 = v17;
  if ( v17 + 1 > v18 )
  {
    v24 = v6 + 8;
    if ( v20 > (unsigned __int64)&v28 )
    {
      v25 = v15;
    }
    else
    {
      v25 = v15;
      if ( (unsigned __int64)&v28 < v20 + 8 * v17 )
      {
        v27 = (char *)&v28 - v20;
        sub_31878D0(v24, v21, v17, v20, v15, (__int64)&v28 - v20);
        v20 = *(_QWORD *)(v6 + 8);
        v17 = *(unsigned int *)(v6 + 16);
        v15 = v25;
        v22 = *(_DWORD *)(v6 + 16);
        v19 = (__int64 *)&v27[v20];
        goto LABEL_16;
      }
    }
    sub_31878D0(v24, v21, v17, v20, v15, (__int64)&v28);
    v17 = *(unsigned int *)(v6 + 16);
    v20 = *(_QWORD *)(v6 + 8);
    v19 = &v28;
    v15 = v25;
    v22 = *(_DWORD *)(v6 + 16);
  }
LABEL_16:
  v23 = (__int64 *)(v20 + 8 * v17);
  if ( v23 )
  {
    *v23 = *v19;
    *v19 = 0;
    v15 = v28;
    v22 = *(_DWORD *)(v6 + 16);
  }
  *(_DWORD *)(v6 + 16) = v22 + 1;
  if ( v15 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 24LL))(v15);
  v6 = a1[3];
LABEL_2:
  sub_31871E0(v6, (__int64)&v29, a3);
  v8 = a1[2];
  v9 = *(_QWORD *)(a3 + 16);
  if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
    v10 = *(_QWORD *)(v8 - 8);
  else
    v10 = v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
  result = 32 * v5 + v10;
  if ( *(_QWORD *)result )
  {
    v12 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = v9;
  if ( v9 )
  {
    v13 = *(_QWORD *)(v9 + 16);
    *(_QWORD *)(result + 8) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = result + 8;
    *(_QWORD *)(result + 16) = v9 + 16;
    *(_QWORD *)(v9 + 16) = result;
  }
  return result;
}
