// Function: sub_22555B0
// Address: 0x22555b0
//
__int64 *__fastcall sub_22555B0(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v8; // rax
  __int128 *v11; // rax
  _DWORD *v12; // rax
  _QWORD *v13; // rax
  char *v14; // rcx
  _QWORD *v15; // r15
  __int64 v16; // rax
  int v17; // eax
  const wchar_t *v18; // rdx
  void *v19; // rsp
  const wchar_t *v20; // rcx
  __int64 v21; // rax
  _DWORD *v22; // r11
  size_t v23; // rax
  void *v24; // rsp
  _BYTE **v25; // [rsp+0h] [rbp-80h] BYREF
  _DWORD *v26; // [rsp+8h] [rbp-78h]
  char *v27; // [rsp+10h] [rbp-70h]
  char *msgid; // [rsp+18h] [rbp-68h]
  char *domainname; // [rsp+20h] [rbp-60h]
  _QWORD *v30; // [rsp+28h] [rbp-58h]
  char v31; // [rsp+38h] [rbp-48h] BYREF
  _BYTE *v32; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v33[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = a1 + 2;
  v8 = *(_QWORD *)(a6 + 8);
  if ( a3 >= 0 && v8 )
  {
    v11 = sub_2254490();
    v12 = sub_22543D0((pthread_mutex_t *)v11, a3);
    if ( !v12 )
      goto LABEL_8;
    v26 = v12;
    v13 = (_QWORD *)sub_22435B0((_QWORD *)v12 + 2, (unsigned int)a3);
    v14 = *(char **)(a6 + 8);
    v33[0] = 0;
    v15 = v13;
    v30 = v33;
    v16 = *v13;
    domainname = v14;
    v17 = (*(__int64 (__fastcall **)(_QWORD *))(v16 + 64))(v15);
    v18 = *(const wchar_t **)a6;
    v27 = &v31;
    v25 = &v32;
    v19 = alloca((_QWORD)domainname * v17 + 9LL);
    v20 = &v18[*(_QWORD *)(a6 + 8)];
    v21 = *v15;
    msgid = (char *)&v25;
    (*(void (__fastcall **)(_QWORD *, _QWORD *, const wchar_t *, const wchar_t *))(v21 + 16))(v15, v33, v18, v20);
    v22 = v26;
    *v32 = 0;
    domainname = (char *)*((_QWORD *)v22 + 1);
    __uselocale();
    domainname = dgettext(domainname, msgid);
    __uselocale();
    if ( msgid == domainname )
    {
LABEL_8:
      *a1 = (__int64)v6;
      sub_2252030(a1, *(const wchar_t **)a6, *(_QWORD *)a6 + 4LL * *(_QWORD *)(a6 + 8));
    }
    else
    {
      msgid = v27;
      v33[0] = 0;
      v23 = strlen(domainname);
      v24 = alloca(4 * v23 + 12);
      (*(void (__fastcall **)(_QWORD *, _QWORD *, char *, char *, char *, _BYTE ***, char *, _BYTE **))(*v15 + 32LL))(
        v15,
        v30,
        domainname,
        &domainname[v23],
        msgid,
        &v25,
        (char *)&v25 + 4 * v23,
        v25);
      *a1 = (__int64)v6;
      sub_2252030(a1, (const wchar_t *)&v25, (__int64)v32);
    }
  }
  else
  {
    *a1 = (__int64)v6;
    sub_2252030(a1, *(const wchar_t **)a6, *(_QWORD *)a6 + 4 * v8);
  }
  return a1;
}
