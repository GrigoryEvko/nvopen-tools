// Function: sub_11DB650
// Address: 0x11db650
//
unsigned __int64 __fastcall sub_11DB650(__int64 a1, __int64 a2, char a3, __int64 *a4, char a5)
{
  __int64 v5; // r14
  unsigned __int64 result; // rax
  __int64 i; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  const char *v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // eax
  _QWORD *v16; // rdi
  unsigned int v17; // r12d
  __int64 v18; // rax
  __int64 v19; // r12
  _QWORD *v20; // rdi
  __int64 **v21; // rax
  bool v22; // zf
  __int64 v23; // rax
  const char *v24; // rax
  __int64 v25; // rdx
  const char *v26; // rcx
  unsigned __int64 v27; // r11
  int v28; // eax
  __int64 v29; // rdi
  __int64 v30; // [rsp+8h] [rbp-A8h]
  char *v31; // [rsp+10h] [rbp-A0h]
  char v32; // [rsp+1Dh] [rbp-93h]
  char v33; // [rsp+1Eh] [rbp-92h]
  char v34; // [rsp+1Fh] [rbp-91h]
  size_t n; // [rsp+20h] [rbp-90h]
  unsigned __int64 na; // [rsp+20h] [rbp-90h]
  int s2; // [rsp+28h] [rbp-88h]
  const char *s2a; // [rsp+28h] [rbp-88h]
  __int64 v39; // [rsp+30h] [rbp-80h] BYREF
  __int64 v40; // [rsp+38h] [rbp-78h]
  __int64 v41; // [rsp+40h] [rbp-70h] BYREF
  __int64 v42; // [rsp+48h] [rbp-68h]
  __int64 v43[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v44; // [rsp+70h] [rbp-40h]

  v5 = *(_QWORD *)(a1 - 32);
  if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v5 + 24) || *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 3 )
    return 0;
  if ( a5 )
  {
    for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(i + 8) )
    {
      v11 = *(_QWORD *)(i + 24);
      if ( *(_BYTE *)v11 != 74 || *(_BYTE *)(*(_QWORD *)(v11 + 8) + 8LL) != 2 )
        return 0;
    }
  }
  v12 = sub_11DB1C0(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v41 = v12;
  if ( a3 )
  {
    v12 = sub_11DB1C0(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))));
    v42 = v12;
    if ( !v41 )
      return 0;
  }
  else
  {
    v42 = 0;
  }
  if ( !v12 )
    return 0;
  v13 = sub_BD5D20(v5);
  if ( (*(_BYTE *)(v5 + 33) & 0x20) != 0 )
  {
    s2 = *(_DWORD *)(a2 + 104);
    n = *(_QWORD *)(a2 + 96);
    v34 = *(_BYTE *)(a2 + 108);
    v33 = *(_BYTE *)(a2 + 109);
    v32 = *(_BYTE *)(a2 + 110);
    v15 = sub_B45210(a1);
    v16 = *(_QWORD **)(a2 + 72);
    *(_DWORD *)(a2 + 104) = v15;
    v17 = *(_DWORD *)(v5 + 36);
    HIDWORD(v40) = 0;
    v44 = 257;
    v39 = sub_BCB160(v16);
    if ( a3 )
      v18 = sub_B33D10(a2, v17, (__int64)&v39, 1, (int)&v41, 2, (unsigned int)v40, (__int64)v43);
    else
      v18 = sub_B33D10(a2, v17, (__int64)&v39, 1, (int)&v41, 1, (unsigned int)v40, (__int64)v43);
    v19 = v18;
    goto LABEL_18;
  }
  na = v14;
  s2a = v13;
  v23 = sub_B43CB0(a1);
  v24 = sub_BD5D20(v23);
  v26 = s2a;
  v27 = na;
  if ( v25 )
  {
    if ( v24[v25 - 1] == 102 && na + 1 == v25 && na <= na + 1 )
    {
      if ( !na )
        return 0;
      v28 = memcmp(v24, s2a, na);
      v26 = s2a;
      v27 = na;
      if ( !v28 )
        return 0;
    }
  }
  v30 = v27;
  v31 = (char *)v26;
  s2 = *(_DWORD *)(a2 + 104);
  n = *(_QWORD *)(a2 + 96);
  v34 = *(_BYTE *)(a2 + 108);
  v33 = *(_BYTE *)(a2 + 109);
  v32 = *(_BYTE *)(a2 + 110);
  v29 = v41;
  *(_DWORD *)(a2 + 104) = sub_B45210(a1);
  v43[0] = *(_QWORD *)(v5 + 120);
  if ( a3 )
    v19 = sub_11CD090(v29, v42, a4, v31, v30, a2, v43);
  else
    v19 = sub_11CC9B0(v29, a4, v31, v30, a2, v43);
LABEL_18:
  v20 = *(_QWORD **)(a2 + 72);
  v44 = 257;
  v21 = (__int64 **)sub_BCB170(v20);
  HIDWORD(v39) = 0;
  v22 = *(_BYTE *)(a2 + 108) == 0;
  v40 = (unsigned int)v39;
  if ( v22 )
    result = sub_11DB4B0((__int64 *)a2, 0x2Eu, v19, v21, (__int64)v43, 0, v39, 0);
  else
    result = sub_B358C0(a2, 0x6Eu, v19, (__int64)v21, (unsigned int)v39, (__int64)v43, 0, 0, 0);
  *(_DWORD *)(a2 + 104) = s2;
  *(_BYTE *)(a2 + 108) = v34;
  *(_QWORD *)(a2 + 96) = n;
  *(_BYTE *)(a2 + 110) = v32;
  *(_BYTE *)(a2 + 109) = v33;
  return result;
}
