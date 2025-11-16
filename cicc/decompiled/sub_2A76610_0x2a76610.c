// Function: sub_2A76610
// Address: 0x2a76610
//
__int64 __fastcall sub_2A76610(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  int v6; // eax
  __int64 v7; // r15
  unsigned __int8 **v8; // rcx
  __int64 v9; // r13
  unsigned __int8 *v10; // rcx
  unsigned int v11; // eax
  unsigned int v12; // esi
  unsigned __int64 v13; // rcx
  unsigned int v15; // r8d
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // rax
  bool v19; // r13
  __int64 v20; // r9
  unsigned __int8 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v25; // rdx
  int v26; // eax
  _QWORD *v27; // rdi
  __int64 *v28; // rdi
  __int64 *v29; // rax
  unsigned __int64 *v30; // r13
  unsigned __int64 *v31; // rdi
  int v32; // ebx
  __int64 *v33; // [rsp+8h] [rbp-68h]
  __int64 *v34; // [rsp+8h] [rbp-68h]
  unsigned __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 *v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  unsigned int v38; // [rsp+18h] [rbp-58h]
  __int64 *v39; // [rsp+18h] [rbp-58h]
  unsigned int v40; // [rsp+18h] [rbp-58h]
  unsigned __int64 v41; // [rsp+20h] [rbp-50h] BYREF
  __int64 v42; // [rsp+28h] [rbp-48h]
  _QWORD v43[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *a2;
  if ( v6 != 48 )
  {
    v7 = 0;
    if ( v6 != 55 )
      return v7;
  }
  if ( (a2[7] & 0x40) != 0 )
    v8 = (unsigned __int8 **)*((_QWORD *)a2 - 1);
  else
    v8 = (unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( *v8 != a3 )
    return 0;
  v9 = (__int64)v8[4];
  if ( *(_BYTE *)v9 != 17 || (unsigned int)*a3 - 42 > 0x11 )
    return 0;
  v10 = (a3[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a3 - 1) : &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
  if ( **((_BYTE **)v10 + 4) != 17 )
    return 0;
  v7 = *(_QWORD *)v10;
  if ( *a2 == 55 )
  {
    v38 = *(_DWORD *)(v9 + 32);
    v11 = *(_DWORD *)(*((_QWORD *)a2 + 1) + 8LL) >> 8;
    v12 = v11;
    if ( v38 > 0x40 )
    {
      v35 = v11;
      v12 = v11;
      if ( v38 - (unsigned int)sub_C444A0(v9 + 24) > 0x40 )
        return 0;
      v13 = **(_QWORD **)(v9 + 24);
      if ( v35 <= v13 )
        return 0;
    }
    else
    {
      v13 = *(_QWORD *)(v9 + 24);
      if ( v11 <= v13 )
        return 0;
    }
    v15 = v13;
    LODWORD(v42) = v12;
    v16 = 1LL << v13;
    v17 = 1LL << v13;
    if ( v12 > 0x40 )
    {
      v40 = v15;
      v37 = v16;
      sub_C43690((__int64)&v41, 0, 0);
      v17 = v37;
      if ( (unsigned int)v42 > 0x40 )
      {
        *(_QWORD *)(v41 + 8LL * (v40 >> 6)) |= v37;
        goto LABEL_23;
      }
    }
    else
    {
      v41 = 0;
    }
    v41 |= v17;
LABEL_23:
    v18 = (__int64 *)sub_BD5C60((__int64)a2);
    v9 = sub_ACCFD0(v18, (__int64)&v41);
    if ( (unsigned int)v42 > 0x40 && v41 )
      j_j___libc_free_0_0(v41);
  }
  v39 = sub_DD8400(*(_QWORD *)(a1 + 16), v7);
  v33 = sub_DD8400(*(_QWORD *)(a1 + 16), v9);
  v36 = (__int64 *)sub_DCB270(*(_QWORD *)(a1 + 16), (__int64)v39, (__int64)v33);
  v19 = sub_B44E60((__int64)a2);
  if ( v19 )
  {
    v28 = *(__int64 **)(a1 + 16);
    v43[1] = v33;
    v43[0] = v36;
    v41 = (unsigned __int64)v43;
    v42 = 0x200000002LL;
    v29 = sub_DC8BD0(v28, (__int64)&v41, 0, 0);
    if ( (_QWORD *)v41 != v43 )
    {
      v34 = v29;
      _libc_free(v41);
      v29 = v34;
    }
    v19 = v39 != v29;
  }
  if ( !sub_D97040(*(_QWORD *)(a1 + 16), *((_QWORD *)a2 + 1)) || v36 != sub_DD8400(*(_QWORD *)(a1 + 16), (__int64)a2) )
    return 0;
  if ( (a2[7] & 0x40) != 0 )
    v21 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v21 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( *(_QWORD *)v21 )
  {
    v22 = *((_QWORD *)v21 + 1);
    **((_QWORD **)v21 + 2) = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = *((_QWORD *)v21 + 2);
  }
  *(_QWORD *)v21 = v7;
  if ( v7 )
  {
    v23 = *(_QWORD *)(v7 + 16);
    *((_QWORD *)v21 + 1) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = v21 + 8;
    *((_QWORD *)v21 + 2) = v7 + 16;
    *(_QWORD *)(v7 + 16) = v21;
  }
  if ( v19 )
    sub_B44F30(a2);
  *(_BYTE *)(a1 + 56) = 1;
  if ( !*((_QWORD *)a3 + 2) )
  {
    v24 = *(_QWORD *)(a1 + 48);
    v25 = *(unsigned int *)(v24 + 8);
    v26 = v25;
    if ( *(_DWORD *)(v24 + 12) <= (unsigned int)v25 )
    {
      v30 = (unsigned __int64 *)sub_C8D7D0(v24, v24 + 16, 0, 0x18u, &v41, v20);
      v31 = &v30[3 * *(unsigned int *)(v24 + 8)];
      if ( v31 )
      {
        *v31 = 6;
        v31[1] = 0;
        v31[2] = (unsigned __int64)a3;
        if ( a3 != (unsigned __int8 *)-8192LL && a3 != (unsigned __int8 *)-4096LL )
          sub_BD73F0((__int64)v31);
      }
      sub_F17F80(v24, v30);
      v32 = v41;
      if ( v24 + 16 != *(_QWORD *)v24 )
        _libc_free(*(_QWORD *)v24);
      ++*(_DWORD *)(v24 + 8);
      *(_QWORD *)v24 = v30;
      *(_DWORD *)(v24 + 12) = v32;
    }
    else
    {
      v27 = (_QWORD *)(*(_QWORD *)v24 + 24 * v25);
      if ( v27 )
      {
        *v27 = 6;
        v27[1] = 0;
        v27[2] = a3;
        if ( a3 != (unsigned __int8 *)-8192LL && a3 != (unsigned __int8 *)-4096LL )
          sub_BD73F0((__int64)v27);
        v26 = *(_DWORD *)(v24 + 8);
      }
      *(_DWORD *)(v24 + 8) = v26 + 1;
    }
  }
  return v7;
}
