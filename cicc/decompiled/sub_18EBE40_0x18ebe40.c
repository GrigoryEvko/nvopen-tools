// Function: sub_18EBE40
// Address: 0x18ebe40
//
__int64 __fastcall sub_18EBE40(unsigned __int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // r8
  __int64 v10; // r15
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r9
  int v16; // esi
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // eax
  unsigned __int64 *i; // r15
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+28h] [rbp-38h]

  v5 = a1[1];
  v6 = *a1;
  v7 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)*a1) >> 5);
  if ( v7 == 0xCCCCCCCCCCCCCCLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = (unsigned __int64 *)a2;
  if ( v7 )
    v8 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v5 - (char *)v6) >> 5);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x3333333333333333LL * (((char *)v5 - (char *)v6) >> 5);
  v13 = a2 - (_QWORD)v6;
  if ( v11 )
  {
    v23 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v12 )
    {
      v27 = 0;
      v14 = 160;
      v32 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0xCCCCCCCCCCCCCCLL )
      v12 = 0xCCCCCCCCCCCCCCLL;
    v23 = 160 * v12;
  }
  v25 = a3;
  v24 = sub_22077B0(v23);
  v13 = a2 - (_QWORD)v6;
  v9 = (unsigned __int64 *)a2;
  v32 = v24;
  a3 = v25;
  v27 = v24 + v23;
  v14 = v24 + 160;
LABEL_7:
  v15 = v32 + v13;
  if ( v15 )
  {
    a4 = *(unsigned int *)(a3 + 8);
    *(_QWORD *)v15 = v15 + 16;
    *(_QWORD *)(v15 + 8) = 0x800000000LL;
    if ( (_DWORD)a4 )
    {
      v26 = v9;
      v29 = a3;
      v31 = v15;
      sub_18E63F0(v15, (char **)a3, a3, a4, (int)v9, v15);
      v9 = v26;
      a3 = v29;
      v15 = v31;
    }
    v16 = *(_DWORD *)(a3 + 152);
    *(_QWORD *)(v15 + 144) = *(_QWORD *)(a3 + 144);
    *(_DWORD *)(v15 + 152) = v16;
  }
  if ( v9 != v6 )
  {
    v17 = v32;
    v18 = (__int64)v6;
    while ( 1 )
    {
      if ( v17 )
      {
        *(_DWORD *)(v17 + 8) = 0;
        *(_QWORD *)v17 = v17 + 16;
        *(_DWORD *)(v17 + 12) = 8;
        a3 = *(unsigned int *)(v18 + 8);
        if ( (_DWORD)a3 )
        {
          v28 = v9;
          v30 = v18;
          sub_18E6310(v17, v18, a3, a4, (int)v9, v15);
          v9 = v28;
          v18 = v30;
        }
        *(_QWORD *)(v17 + 144) = *(_QWORD *)(v18 + 144);
        *(_DWORD *)(v17 + 152) = *(_DWORD *)(v18 + 152);
      }
      v18 += 160;
      if ( v9 == (unsigned __int64 *)v18 )
        break;
      v17 += 160;
    }
    v14 = v17 + 320;
  }
  if ( v9 != v5 )
  {
    do
    {
      *(_DWORD *)(v14 + 8) = 0;
      *(_QWORD *)v14 = v14 + 16;
      v20 = *(_DWORD *)(v10 + 8);
      *(_DWORD *)(v14 + 12) = 8;
      if ( v20 )
        sub_18E6310(v14, v10, a3, a4, (int)v9, v15);
      v19 = *(_QWORD *)(v10 + 144);
      v10 += 160;
      v14 += 160;
      *(_QWORD *)(v14 - 16) = v19;
      *(_DWORD *)(v14 - 8) = *(_DWORD *)(v10 - 8);
    }
    while ( v5 != (unsigned __int64 *)v10 );
  }
  for ( i = v6; v5 != i; i += 20 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      _libc_free(*i);
  }
  if ( v6 )
    j_j___libc_free_0(v6, (char *)a1[2] - (char *)v6);
  a1[1] = (unsigned __int64 *)v14;
  *a1 = (unsigned __int64 *)v32;
  a1[2] = (unsigned __int64 *)v27;
  return v27;
}
