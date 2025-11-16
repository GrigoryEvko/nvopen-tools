// Function: sub_22559E0
// Address: 0x22559e0
//
__int64 *__fastcall sub_22559E0(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int128 *v9; // rax
  _DWORD *v10; // rax
  __int64 v11; // r14
  __int64 v12; // r15
  int v13; // eax
  __int64 v14; // rdx
  __int64 v15; // rsi
  void *v16; // rsp
  __int64 v17; // r11
  char *v18; // rbx
  size_t v19; // rax
  void *v20; // rsp
  __int64 v21; // rbx
  unsigned __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r14
  void *v25; // rcx
  wchar_t v26[2]; // [rsp+0h] [rbp-80h] BYREF
  wchar_t **v27; // [rsp+8h] [rbp-78h]
  _BYTE *v28; // [rsp+10h] [rbp-70h]
  char *domainname; // [rsp+18h] [rbp-68h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  _QWORD *v31; // [rsp+28h] [rbp-58h]
  _BYTE v32[8]; // [rsp+38h] [rbp-48h] BYREF
  wchar_t *v33; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v34[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( a3 < 0 )
    goto LABEL_3;
  if ( !*(_QWORD *)(*a6 - 24) )
    goto LABEL_3;
  v9 = sub_2254490();
  v10 = sub_22543D0((pthread_mutex_t *)v9, a3);
  if ( !v10 )
    goto LABEL_3;
  v30 = (__int64)v10;
  v34[0] = 0;
  v11 = sub_22435B0((_QWORD *)v10 + 2, (unsigned int)a3);
  v31 = v34;
  v12 = *(_QWORD *)(*a6 - 24);
  v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 64LL))(v11);
  v14 = *a6;
  v28 = v32;
  v15 = v12 * v13;
  v27 = &v33;
  v16 = alloca(v15 + 9);
  (*(void (__fastcall **)(__int64, _QWORD *, __int64, __int64, _BYTE *, wchar_t *, char *, wchar_t **))(*(_QWORD *)v11 + 16LL))(
    v11,
    v34,
    v14,
    v14 + 4LL * *(_QWORD *)(v14 - 24),
    v32,
    v26,
    (char *)v26 + v15,
    &v33);
  v17 = v30;
  *(_BYTE *)v33 = 0;
  domainname = *(char **)(v17 + 8);
  v30 = __uselocale();
  v18 = dgettext(domainname, (const char *)v26);
  __uselocale();
  if ( v26 != (wchar_t *)v18 )
  {
    v30 = (__int64)v28;
    v34[0] = 0;
    v19 = strlen(v18);
    v20 = alloca(4 * v19 + 12);
    (*(void (__fastcall **)(__int64, _QWORD *, char *, char *, __int64, wchar_t *))(*(_QWORD *)v11 + 32LL))(
      v11,
      v31,
      v18,
      &v18[v19],
      v30,
      v26);
    if ( v33 == v26 )
    {
      v25 = &unk_4FD67F8;
    }
    else
    {
      v21 = (char *)v33 - (char *)v26;
      v22 = v33 - v26;
      v23 = sub_2216040(v22, 0);
      v24 = v23;
      v25 = (void *)(v23 + 24);
      if ( v22 == 1 )
      {
        *(_DWORD *)(v23 + 24) = v26[0];
      }
      else if ( v22 )
      {
        v31 = (_QWORD *)(v23 + 24);
        wmemcpy((wchar_t *)(v23 + 24), v26, v21 >> 2);
        v25 = v31;
      }
      if ( (_UNKNOWN *)v24 != &unk_4FD67E0 )
      {
        *(_DWORD *)(v24 + 16) = 0;
        *(_QWORD *)v24 = v22;
        *(_DWORD *)(v24 + v21 + 24) = 0;
      }
    }
    *a1 = (__int64)v25;
  }
  else
  {
LABEL_3:
    sub_2216B00(a1, a6);
  }
  return a1;
}
