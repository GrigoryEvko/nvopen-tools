// Function: sub_2BDF0D0
// Address: 0x2bdf0d0
//
unsigned __int64 *__fastcall sub_2BDF0D0(__int64 a1)
{
  unsigned __int8 *v1; // rax
  __int64 v2; // r12
  unsigned int v3; // r15d
  __int64 v4; // r13
  unsigned __int8 v5; // r14
  unsigned __int8 v6; // cl
  __int64 (__fastcall *v7)(__int64, unsigned int); // rax
  char *v8; // rax
  char v9; // dl
  __int64 v10; // rdx
  unsigned __int64 *result; // rax
  int v12; // r13d
  __int64 v13; // rsi
  size_t v14; // r12
  char v15; // r10
  __int64 v16; // rax
  unsigned __int64 v17; // r9
  unsigned __int64 v18; // rdx
  char v19; // [rsp+Fh] [rbp-31h]

  v1 = *(unsigned __int8 **)(a1 + 176);
  v2 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 176) = v1 + 1;
  v3 = (char)*v1;
  v4 = *v1;
  v5 = *v1;
  v6 = *(_BYTE *)(v2 + v4 + 313);
  if ( !v6 )
  {
    v6 = *v1;
    v7 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v2 + 64LL);
    if ( v7 != sub_2216C50 )
      v6 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v7)(v2, v3, 0, v3);
    if ( v6 )
      *(_BYTE *)(v2 + v4 + 313) = v6;
  }
  v8 = *(char **)(a1 + 152);
  v9 = *v8;
  if ( *v8 )
  {
    while ( v6 != v9 )
    {
      v9 = v8[2];
      v8 += 2;
      if ( !v9 )
        goto LABEL_11;
    }
    v10 = *(_QWORD *)(a1 + 208);
    *(_DWORD *)(a1 + 144) = 1;
    return sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v10, 1u, v8[1]);
  }
  else
  {
LABEL_11:
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2LL * v5 + 1) & 8) == 0
      || (unsigned __int8)(v5 - 56) <= 1u )
    {
      abort();
    }
    v12 = 2;
    sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, *(_QWORD *)(a1 + 208), 1u, v3);
    while ( 1 )
    {
      result = *(unsigned __int64 **)(a1 + 176);
      if ( result == *(unsigned __int64 **)(a1 + 184) )
        break;
      v13 = *(unsigned __int8 *)result;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2 * v13 + 1) & 8) == 0
        || (unsigned __int8)(v13 - 56) <= 1u )
      {
        break;
      }
      v14 = *(_QWORD *)(a1 + 208);
      *(_QWORD *)(a1 + 176) = (char *)result + 1;
      v15 = *(_BYTE *)result;
      v16 = *(_QWORD *)(a1 + 200);
      v17 = v14 + 1;
      v18 = v16 == a1 + 216 ? 15LL : *(_QWORD *)(a1 + 216);
      if ( v17 > v18 )
      {
        v19 = v15;
        sub_2240BB0((unsigned __int64 *)(a1 + 200), v14, 0, 0, 1u);
        v16 = *(_QWORD *)(a1 + 200);
        v17 = v14 + 1;
        v15 = v19;
      }
      *(_BYTE *)(v16 + v14) = v15;
      result = *(unsigned __int64 **)(a1 + 200);
      *(_QWORD *)(a1 + 208) = v17;
      *((_BYTE *)result + v14 + 1) = 0;
      if ( v12 == 1 )
        break;
      v12 = 1;
    }
    *(_DWORD *)(a1 + 144) = 2;
  }
  return result;
}
