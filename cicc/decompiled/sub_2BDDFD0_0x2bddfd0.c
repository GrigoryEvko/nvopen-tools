// Function: sub_2BDDFD0
// Address: 0x2bddfd0
//
unsigned __int64 *__fastcall sub_2BDDFD0(__int64 a1)
{
  unsigned __int8 *v1; // rax
  __int64 v2; // r13
  unsigned int v4; // r15d
  __int64 v5; // r14
  char v6; // cl
  __int64 (__fastcall *v7)(__int64, unsigned int); // rax
  char *v8; // rdx
  char v9; // al
  __int64 v10; // rdx
  unsigned __int64 *result; // rax
  __int64 v12; // r9
  unsigned __int64 *v13; // rdi
  char v14; // r8
  __int64 v15; // rdx
  __int64 i; // r9
  __int64 v17; // rax
  size_t v18; // r12
  char v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  char *v25; // rax
  __int64 v26; // rdx
  _BYTE *v27; // rax
  __int64 v28; // r9
  int v29; // r13d
  int v30; // r10d
  unsigned __int8 *v31; // rax
  size_t v32; // r12
  unsigned __int8 v33; // r11
  __int64 v34; // rax
  unsigned __int64 v35; // r15
  unsigned __int64 v36; // rdx
  __int64 v37; // [rsp+8h] [rbp-48h]
  int v38; // [rsp+14h] [rbp-3Ch]
  __int64 v39; // [rsp+18h] [rbp-38h]
  unsigned __int8 v40; // [rsp+18h] [rbp-38h]

  v1 = *(unsigned __int8 **)(a1 + 176);
  if ( v1 == *(unsigned __int8 **)(a1 + 184) )
    goto LABEL_49;
  v2 = *(_QWORD *)(a1 + 192);
  *(_QWORD *)(a1 + 176) = v1 + 1;
  v4 = (char)*v1;
  v5 = *v1;
  v6 = *(_BYTE *)(v2 + v5 + 313);
  if ( !v6 )
  {
    v6 = v4;
    v7 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v2 + 64LL);
    if ( v7 != sub_2216C50 )
      v6 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v7)(v2, v4, 0, v4);
    if ( v6 )
      *(_BYTE *)(v2 + v5 + 313) = v6;
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
        goto LABEL_16;
    }
    if ( (_BYTE)v4 != 98 || *(_DWORD *)(a1 + 136) == 2 )
    {
      v12 = *(_QWORD *)(a1 + 208);
      v13 = (unsigned __int64 *)(a1 + 200);
      *(_DWORD *)(a1 + 144) = 1;
      v14 = v8[1];
      v15 = v12;
      return sub_2240FD0(v13, 0, v15, 1u, v14);
    }
LABEL_13:
    v10 = *(_QWORD *)(a1 + 208);
    *(_DWORD *)(a1 + 144) = 24;
    return sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v10, 1u, 112);
  }
LABEL_16:
  if ( (_BYTE)v4 == 98 )
    goto LABEL_13;
  if ( (_BYTE)v4 == 66 )
  {
    v24 = *(_QWORD *)(a1 + 208);
    *(_DWORD *)(a1 + 144) = 24;
    return sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v24, 1u, 110);
  }
  if ( (unsigned __int8)(v4 - 68) <= 0x33u )
  {
    v23 = 0x8800100088001LL;
    if ( _bittest64(&v23, v4 - 68) )
    {
      *(_DWORD *)(a1 + 144) = 14;
      return sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, *(_QWORD *)(a1 + 208), 1u, v4);
    }
    if ( (_BYTE)v4 == 99 )
    {
      v25 = *(char **)(a1 + 176);
      if ( v25 != *(char **)(a1 + 184) )
      {
        *(_QWORD *)(a1 + 176) = v25 + 1;
        v26 = *(_QWORD *)(a1 + 208);
        *(_DWORD *)(a1 + 144) = 1;
        return sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v26, 1u, *v25);
      }
LABEL_49:
      abort();
    }
  }
  if ( (_BYTE)v4 != 120 && (_BYTE)v4 != 117 )
  {
    v15 = *(_QWORD *)(a1 + 208);
    if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2LL * (unsigned __int8)v4 + 1) & 8) != 0 )
    {
      sub_2240FD0((unsigned __int64 *)(a1 + 200), 0, v15, 1u, v4);
      result = *(unsigned __int64 **)(a1 + 176);
      for ( i = a1 + 216; result != *(unsigned __int64 **)(a1 + 184); result = *(unsigned __int64 **)(a1 + 176) )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2LL * *(unsigned __int8 *)result + 1) & 8) == 0 )
          break;
        v18 = *(_QWORD *)(a1 + 208);
        *(_QWORD *)(a1 + 176) = (char *)result + 1;
        v19 = *(_BYTE *)result;
        v20 = *(_QWORD *)(a1 + 200);
        v21 = v18 + 1;
        v22 = v20 == i ? 15LL : *(_QWORD *)(a1 + 216);
        if ( v21 > v22 )
        {
          v39 = i;
          sub_2240BB0((unsigned __int64 *)(a1 + 200), v18, 0, 0, 1u);
          v20 = *(_QWORD *)(a1 + 200);
          i = v39;
        }
        *(_BYTE *)(v20 + v18) = v19;
        v17 = *(_QWORD *)(a1 + 200);
        *(_QWORD *)(a1 + 208) = v21;
        *(_BYTE *)(v17 + v18 + 1) = 0;
      }
      *(_DWORD *)(a1 + 144) = 4;
      return result;
    }
    v14 = v4;
    v13 = (unsigned __int64 *)(a1 + 200);
    *(_DWORD *)(a1 + 144) = 1;
    return sub_2240FD0(v13, 0, v15, 1u, v14);
  }
  v27 = *(_BYTE **)(a1 + 200);
  v28 = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0;
  *v27 = 0;
  v29 = 0;
  v30 = 2 * ((_BYTE)v4 != 120) + 2;
  do
  {
    v31 = *(unsigned __int8 **)(a1 + 176);
    if ( v31 == *(unsigned __int8 **)(a1 + 184)
      || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 192) + 48LL) + 2LL * *v31 + 1) & 0x10) == 0 )
    {
      goto LABEL_49;
    }
    v32 = *(_QWORD *)(a1 + 208);
    *(_QWORD *)(a1 + 176) = v31 + 1;
    v33 = *v31;
    v34 = *(_QWORD *)(a1 + 200);
    v35 = v32 + 1;
    if ( v34 == v28 )
      v36 = 15;
    else
      v36 = *(_QWORD *)(a1 + 216);
    if ( v35 > v36 )
    {
      v37 = v28;
      v38 = v30;
      v40 = v33;
      sub_2240BB0((unsigned __int64 *)(a1 + 200), v32, 0, 0, 1u);
      v34 = *(_QWORD *)(a1 + 200);
      v28 = v37;
      v30 = v38;
      v33 = v40;
    }
    *(_BYTE *)(v34 + v32) = v33;
    result = *(unsigned __int64 **)(a1 + 200);
    ++v29;
    *(_QWORD *)(a1 + 208) = v35;
    *((_BYTE *)result + v32 + 1) = 0;
  }
  while ( v30 != v29 );
  *(_DWORD *)(a1 + 144) = 3;
  return result;
}
