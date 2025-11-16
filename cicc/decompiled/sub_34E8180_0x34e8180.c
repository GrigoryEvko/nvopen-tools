// Function: sub_34E8180
// Address: 0x34e8180
//
__int64 __fastcall sub_34E8180(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r13
  __int64 v7; // rax
  unsigned __int64 v9; // r14
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rcx
  unsigned int v14; // eax
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  const void *v17; // r8
  size_t v18; // r14
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  const void *v22; // r14
  unsigned int v23; // r12d
  size_t v24; // r15
  unsigned __int64 v25; // rdx
  int v26; // eax
  unsigned __int8 v27; // dl
  __int64 result; // rax
  unsigned __int64 v29; // r15
  __int64 *v30; // r12
  __int64 v31; // r13
  __int64 *i; // r14
  char v33; // [rsp+4h] [rbp-6Ch]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  char src; // [rsp+30h] [rbp-40h]
  void *srca; // [rsp+30h] [rbp-40h]
  __int64 v41; // [rsp+38h] [rbp-38h]
  int v42; // [rsp+38h] [rbp-38h]
  __int64 *v43; // [rsp+38h] [rbp-38h]
  const void *v44; // [rsp+38h] [rbp-38h]

  v6 = *(_QWORD **)(*(_QWORD *)(a2 + 16) + 32LL);
  v7 = *(_QWORD *)(a3 + 16);
  v33 = a5;
  src = a5;
  v34 = v7;
  v38 = v7 + 48;
  if ( *(_QWORD *)(v7 + 56) == v7 + 48 )
  {
LABEL_18:
    if ( !v33 )
    {
      v29 = 8LL * *(unsigned int *)(v34 + 120);
      if ( v29 )
      {
        srca = *(void **)(v34 + 112);
        v43 = (__int64 *)sub_22077B0(v29);
        v30 = &v43[v29 / 8];
        memcpy(v43, srca, v29);
      }
      else
      {
        v43 = 0;
        v30 = 0;
      }
      v31 = *(_QWORD *)(v34 + 8);
      if ( v31 == *(_QWORD *)(v34 + 32) + 320LL )
        v31 = 0;
      if ( (*(_BYTE *)a3 & 0x40) == 0 )
        v31 = 0;
      for ( i = v43; v30 != i; ++i )
      {
        if ( v31 != *i )
          sub_2E33F80(*(_QWORD *)(a2 + 16), *i, -1, a4, a5, a6);
      }
      if ( v43 )
        j_j___libc_free_0((unsigned __int64)v43);
    }
  }
  else
  {
    v9 = *(_QWORD *)(v7 + 56);
    v37 = a1 + 224;
    while ( 1 )
    {
      if ( src )
      {
        v10 = *(_DWORD *)(v9 + 44);
        if ( (v10 & 4) != 0 || (v10 & 8) == 0 )
          v11 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 24LL) >> 10) & 1LL;
        else
          LOBYTE(v11) = sub_2E88A90(v9, 1024, 1);
        if ( (_BYTE)v11 )
          break;
      }
      v12 = (__int64)sub_2E7B2C0(v6, v9);
      if ( sub_2E88ED0(v9, 0) )
        sub_2E7E170((__int64)v6, v9, v12);
      v41 = *(_QWORD *)(a2 + 16);
      sub_2E31040((__int64 *)(v41 + 40), v12);
      v13 = *(_QWORD *)(v41 + 48);
      *(_QWORD *)(v12 + 8) = v41 + 48;
      v13 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v12 = v13 | *(_QWORD *)v12 & 7LL;
      *(_QWORD *)(v13 + 8) = v12;
      *(_QWORD *)(v41 + 48) = v12 | *(_QWORD *)(v41 + 48) & 7LL;
      ++*(_DWORD *)(a2 + 4);
      v42 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64))(**(_QWORD **)(a1 + 528) + 1168LL))(
              *(_QWORD *)(a1 + 528),
              v9);
      v14 = sub_2FF8080(v37, v9, 0);
      if ( v14 > 1 )
        *(_DWORD *)(a2 + 8) = v14 + *(_DWORD *)(a2 + 8) - 1;
      *(_DWORD *)(a2 + 12) += v42;
      v15 = *(_QWORD *)(a1 + 528);
      v16 = *(__int64 (**)())(*(_QWORD *)v15 + 920LL);
      if ( (v16 == sub_2DB1B30 || !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v16)(v15, v9))
        && (unsigned __int16)(*(_WORD *)(v12 + 68) - 14) > 4u
        && !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 968LL))(
              *(_QWORD *)(a1 + 528),
              v12,
              *(_QWORD *)a4,
              *(unsigned int *)(a4 + 8)) )
      {
        BUG();
      }
      sub_34E7040(v12, a1 + 560);
      if ( !v9 )
        BUG();
      if ( (*(_BYTE *)v9 & 4) != 0 )
      {
        v9 = *(_QWORD *)(v9 + 8);
        if ( v38 == v9 )
          goto LABEL_18;
      }
      else
      {
        while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
          v9 = *(_QWORD *)(v9 + 8);
        v9 = *(_QWORD *)(v9 + 8);
        if ( v38 == v9 )
          goto LABEL_18;
      }
    }
  }
  v17 = *(const void **)(a3 + 216);
  v18 = 40LL * *(unsigned int *)(a3 + 224);
  v19 = *(unsigned int *)(a3 + 224);
  v20 = *(unsigned int *)(a2 + 224);
  if ( v19 + v20 > (unsigned __int64)*(unsigned int *)(a2 + 228) )
  {
    v44 = *(const void **)(a3 + 216);
    sub_C8D5F0(a2 + 216, (const void *)(a2 + 232), v19 + v20, 0x28u, (__int64)v17, a6);
    v20 = *(unsigned int *)(a2 + 224);
    v17 = v44;
  }
  if ( v18 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 216) + 40 * v20), v17, v18);
    LODWORD(v20) = *(_DWORD *)(a2 + 224);
  }
  LODWORD(v21) = v19 + v20;
  *(_DWORD *)(a2 + 224) = v21;
  v21 = (unsigned int)v21;
  v22 = *(const void **)a4;
  v23 = *(_DWORD *)(a4 + 8);
  v24 = 40LL * v23;
  v25 = v23 + (unsigned __int64)(unsigned int)v21;
  if ( v25 > *(unsigned int *)(a2 + 228) )
  {
    sub_C8D5F0(a2 + 216, (const void *)(a2 + 232), v25, 0x28u, (__int64)v17, a6);
    v21 = *(unsigned int *)(a2 + 224);
  }
  if ( v24 )
  {
    memcpy((void *)(*(_QWORD *)(a2 + 216) + 40 * v21), v22, v24);
    LODWORD(v21) = *(_DWORD *)(a2 + 224);
  }
  *(_DWORD *)(a2 + 224) = v23 + v21;
  v26 = *(unsigned __int8 *)(a2 + 1);
  v27 = *(_BYTE *)(a3 + 1);
  *(_BYTE *)a2 &= ~4u;
  result = ((unsigned __int8)v26 | v27) & 2 | v26 & 0xFFFFFFFD;
  *(_BYTE *)(a2 + 1) = result;
  return result;
}
