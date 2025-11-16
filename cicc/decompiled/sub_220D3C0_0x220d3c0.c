// Function: sub_220D3C0
// Address: 0x220d3c0
//
__int64 __fastcall sub_220D3C0(__int64 a1, __int64 a2)
{
  int v3; // r8d
  _DWORD *v4; // rax
  int v5; // edx
  const char *v6; // r15
  char *v7; // rax
  __int64 v8; // r12
  char v9; // r13
  char *v10; // rdi
  size_t v11; // rax
  const wchar_t *v12; // rdi
  const wchar_t *v13; // rdi
  size_t v14; // rax
  char *v15; // rdi
  size_t v16; // rax
  wchar_t *v17; // r15
  char v18; // r15
  char *v19; // rax
  __int64 v20; // r12
  char *v21; // rax
  __int64 v22; // rbx
  __int64 result; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  char *v26; // rsi
  size_t v27; // rax
  size_t v28; // rax
  char *v29; // rdi
  size_t v30; // r12
  unsigned __int64 v31; // rdi
  wchar_t *v32; // r15
  size_t v33; // r12
  size_t v34; // r12
  unsigned __int64 v35; // rdi
  size_t v36; // r12
  unsigned __int64 v37; // rdi
  __int64 v38; // rax
  char v39; // [rsp+0h] [rbp-68h]
  size_t v40; // [rsp+0h] [rbp-68h]
  wchar_t *v41; // [rsp+0h] [rbp-68h]
  void *v42; // [rsp+8h] [rbp-60h]
  char *s; // [rsp+10h] [rbp-58h] BYREF
  char *src; // [rsp+18h] [rbp-50h] BYREF
  char *v45; // [rsp+20h] [rbp-48h] BYREF
  mbstate_t ps; // [rsp+28h] [rbp-40h] BYREF

  if ( !*(_QWORD *)(a1 + 16) )
  {
    v38 = sub_22077B0(0xA0u);
    *(_DWORD *)(v38 + 8) = 0;
    *(_QWORD *)(v38 + 16) = 0;
    *(_QWORD *)v38 = off_4A048A0;
    *(_QWORD *)(v38 + 24) = 0;
    *(_BYTE *)(v38 + 32) = 0;
    *(_QWORD *)(v38 + 36) = 0;
    *(_QWORD *)(v38 + 48) = 0;
    *(_QWORD *)(v38 + 56) = 0;
    *(_QWORD *)(v38 + 64) = 0;
    *(_QWORD *)(v38 + 72) = 0;
    *(_QWORD *)(v38 + 80) = 0;
    *(_QWORD *)(v38 + 88) = 0;
    *(_QWORD *)(v38 + 96) = 0;
    *(_DWORD *)(v38 + 104) = 0;
    *(_BYTE *)(v38 + 152) = 0;
    *(_QWORD *)(a1 + 16) = v38;
  }
  if ( !a2 )
  {
    v24 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(v24 + 48) = &dword_4360234;
    *(_QWORD *)(v24 + 64) = &dword_4360234;
    *(_QWORD *)(v24 + 80) = &dword_4360234;
    *(_QWORD *)(v24 + 36) = 0x2C0000002ELL;
    *(_QWORD *)(v24 + 16) = byte_3F871B3;
    *(_QWORD *)(v24 + 24) = 0;
    *(_DWORD *)(v24 + 100) = unk_4363345;
    *(_BYTE *)(v24 + 32) = 0;
    *(_QWORD *)(v24 + 56) = 0;
    *(_QWORD *)(v24 + 72) = 0;
    *(_QWORD *)(v24 + 88) = 0;
    *(_DWORD *)(v24 + 96) = 0;
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 104LL) = unk_4363345;
    v25 = *(_QWORD *)(a1 + 16);
    v26 = off_4CDFAD0;
    for ( result = 0; result != 11; ++result )
      *(_DWORD *)(v25 + 4 * result + 108) = v26[result];
    return result;
  }
  __uselocale();
  *(_DWORD *)(*(_QWORD *)(a1 + 16) + 36LL) = __nl_langinfo_l();
  v3 = __nl_langinfo_l();
  v4 = *(_DWORD **)(a1 + 16);
  v5 = v4[9];
  v4[10] = v3;
  if ( v5 )
  {
    *(_DWORD *)(*(_QWORD *)(a1 + 16) + 96LL) = *(char *)__nl_langinfo_l();
  }
  else
  {
    v4[24] = 0;
    v4[9] = 46;
  }
  v6 = (const char *)__nl_langinfo_l();
  s = (char *)__nl_langinfo_l();
  src = (char *)__nl_langinfo_l();
  v45 = (char *)__nl_langinfo_l();
  v7 = (char *)__nl_langinfo_l();
  v8 = *(_QWORD *)(a1 + 16);
  v9 = *v7;
  if ( *(_DWORD *)(v8 + 40) )
  {
    v28 = strlen(v6);
    v40 = v28;
    if ( v28 )
    {
      v33 = v28 + 1;
      v42 = (void *)sub_2207820(v28 + 1);
      memcpy(v42, v6, v33);
      v8 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(v8 + 16) = v42;
    }
    else
    {
      *(_BYTE *)(v8 + 32) = 0;
      *(_QWORD *)(v8 + 16) = byte_3F871B3;
    }
    v29 = s;
    *(_QWORD *)(v8 + 24) = v40;
    v11 = strlen(v29);
    if ( !v11 )
      goto LABEL_8;
LABEL_22:
    ps = 0;
    v30 = v11 + 1;
    v31 = 4 * (v11 + 1);
    if ( v11 + 1 > 0x1FFFFFFFFFFFFFFELL )
      v31 = -1;
    v32 = (wchar_t *)sub_2207820(v31);
    mbsrtowcs(v32, (const char **)&s, v30, &ps);
    v8 = *(_QWORD *)(a1 + 16);
    v12 = v32;
    *(_QWORD *)(v8 + 64) = v32;
    goto LABEL_9;
  }
  *(_BYTE *)(v8 + 32) = 0;
  v10 = s;
  *(_QWORD *)(v8 + 16) = byte_3F871B3;
  *(_QWORD *)(v8 + 24) = 0;
  *(_DWORD *)(v8 + 40) = 44;
  v11 = strlen(v10);
  if ( v11 )
    goto LABEL_22;
LABEL_8:
  v12 = &dword_4360234;
  *(_QWORD *)(v8 + 64) = &dword_4360234;
LABEL_9:
  *(_QWORD *)(v8 + 72) = wcslen(v12);
  if ( v9 )
  {
    v27 = strlen(src);
    if ( v27 )
    {
      ps = 0;
      v34 = v27 + 1;
      v35 = 4 * (v27 + 1);
      if ( v27 + 1 > 0x1FFFFFFFFFFFFFFELL )
        v35 = -1;
      v41 = (wchar_t *)sub_2207820(v35);
      mbsrtowcs(v41, (const char **)&src, v34, &ps);
      v8 = *(_QWORD *)(a1 + 16);
      v13 = v41;
      *(_QWORD *)(v8 + 80) = v41;
    }
    else
    {
      v13 = &dword_4360234;
      *(_QWORD *)(v8 + 80) = &dword_4360234;
    }
  }
  else
  {
    v13 = "(";
    *(_QWORD *)(v8 + 80) = "(";
  }
  v14 = wcslen(v13);
  v15 = v45;
  *(_QWORD *)(v8 + 88) = v14;
  v16 = strlen(v15);
  if ( v16 )
  {
    ps = 0;
    v36 = v16 + 1;
    v37 = 4 * (v16 + 1);
    if ( v16 + 1 > 0x1FFFFFFFFFFFFFFELL )
      v37 = -1;
    v17 = (wchar_t *)sub_2207820(v37);
    mbsrtowcs(v17, (const char **)&v45, v36, &ps);
    v8 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(v8 + 48) = v17;
  }
  else
  {
    v17 = (wchar_t *)&dword_4360234;
    *(_QWORD *)(v8 + 48) = &dword_4360234;
  }
  *(_QWORD *)(v8 + 56) = wcslen(v17);
  v39 = *(_BYTE *)__nl_langinfo_l();
  v18 = *(_BYTE *)__nl_langinfo_l();
  v19 = (char *)__nl_langinfo_l();
  v20 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(v20 + 100) = sub_220C5E0(v39, v18, *v19);
  LOBYTE(v20) = *(_BYTE *)__nl_langinfo_l();
  v21 = (char *)__nl_langinfo_l();
  v22 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(v22 + 104) = sub_220C5E0(v20, *v21, v9);
  return __uselocale();
}
