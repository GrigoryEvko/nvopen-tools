// Function: sub_2428BD0
// Address: 0x2428bd0
//
__int64 __fastcall sub_2428BD0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax
  size_t v4; // r13
  void *v5; // r14
  int v6; // eax
  int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // r15d
  __int64 pw_passwd; // rbx
  const char *pw_name; // r13
  _QWORD *v16; // r15
  _QWORD *v17; // rdx
  _QWORD *v18; // r14
  size_t v19; // rbx
  void *v20; // r13
  int v21; // eax
  unsigned int v22; // r8d
  __int64 *v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // r8d
  __int64 *v27; // rcx
  __int64 v28; // r14
  __int64 *v29; // rdx
  _QWORD *v30; // rdi
  _QWORD *v31; // r14
  _QWORD *v32; // r14
  __int64 *v33; // [rsp+8h] [rbp-238h]
  _QWORD *v34; // [rsp+8h] [rbp-238h]
  __int64 *v35; // [rsp+10h] [rbp-230h]
  _QWORD *v36; // [rsp+18h] [rbp-228h]
  unsigned int v37; // [rsp+18h] [rbp-228h]
  _QWORD *v38; // [rsp+18h] [rbp-228h]
  _QWORD v39[4]; // [rsp+20h] [rbp-220h] BYREF
  __int16 v40; // [rsp+40h] [rbp-200h]
  void *src; // [rsp+50h] [rbp-1F0h] BYREF
  size_t n; // [rsp+58h] [rbp-1E8h]
  char v43; // [rsp+68h] [rbp-1D8h] BYREF
  struct passwd v44[7]; // [rsp+F0h] [rbp-150h] BYREF

  if ( *(_QWORD *)(a1 + 320) == *(_QWORD *)(a1 + 328) && *(_QWORD *)(a1 + 344) == *(_QWORD *)(a1 + 352) )
    return 1;
  v3 = (_BYTE *)sub_B92180(a2);
  sub_2427090(&src, v3);
  v4 = n;
  v5 = src;
  v35 = (__int64 *)(a1 + 400);
  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(a1 + 400), v5, v4, v6);
  if ( v7 == -1
    || (v10 = *(_QWORD *)(a1 + 400), v8 = *(unsigned int *)(a1 + 408), v11 = v10 + 8LL * v7, v11 == v10 + 8 * v8) )
  {
    v44[0].pw_passwd = 0;
    v44[0].pw_name = (char *)&v44[0].pw_gecos;
    v40 = 261;
    *(_QWORD *)&v44[0].pw_uid = 256;
    v39[0] = src;
    v39[1] = n;
    if ( (unsigned int)sub_C84130((__int64)v39, v44, 0, v8, v9) )
    {
      pw_passwd = n;
      pw_name = (const char *)src;
    }
    else
    {
      pw_passwd = (__int64)v44[0].pw_passwd;
      pw_name = v44[0].pw_name;
    }
    v16 = *(_QWORD **)(a1 + 352);
    v17 = *(_QWORD **)(a1 + 344);
    v36 = *(_QWORD **)(a1 + 328);
    if ( v36 == *(_QWORD **)(a1 + 320) )
    {
      v32 = *(_QWORD **)(a1 + 344);
      if ( v17 == v16 )
      {
LABEL_42:
        v12 = 1;
        goto LABEL_16;
      }
      while ( !(unsigned __int8)sub_C89090(v32, pw_name, pw_passwd, 0, 0) )
      {
        v32 += 2;
        if ( v16 == v32 )
          goto LABEL_42;
      }
    }
    else
    {
      v18 = *(_QWORD **)(a1 + 320);
      if ( v17 == v16 )
      {
        v30 = *(_QWORD **)(a1 + 320);
        do
        {
          v34 = v30;
          v12 = sub_C89090(v30, pw_name, pw_passwd, 0, 0);
          if ( (_BYTE)v12 )
            break;
          v30 += 2;
        }
        while ( v36 != v34 + 2 );
        goto LABEL_16;
      }
      while ( 1 )
      {
        v12 = sub_C89090(v18, pw_name, pw_passwd, 0, 0);
        if ( (_BYTE)v12 )
          break;
        v18 += 2;
        if ( v36 == v18 )
          goto LABEL_16;
      }
      v31 = *(_QWORD **)(a1 + 344);
      v38 = *(_QWORD **)(a1 + 352);
      if ( v31 == v38 )
      {
LABEL_16:
        v19 = n;
        v20 = src;
        v21 = sub_C92610();
        v22 = sub_C92740((__int64)v35, v20, v19, v21);
        v23 = (__int64 *)(*(_QWORD *)(a1 + 400) + 8LL * v22);
        v24 = *v23;
        if ( *v23 )
        {
          if ( v24 != -8 )
          {
LABEL_18:
            *(_BYTE *)(v24 + 8) = v12;
            if ( (char **)v44[0].pw_name != &v44[0].pw_gecos )
              _libc_free((unsigned __int64)v44[0].pw_name);
            goto LABEL_5;
          }
          --*(_DWORD *)(a1 + 416);
        }
        v33 = v23;
        v37 = v22;
        v25 = sub_C7D670(v19 + 17, 8);
        v26 = v37;
        v27 = v33;
        v28 = v25;
        if ( v19 )
        {
          memcpy((void *)(v25 + 16), v20, v19);
          v26 = v37;
          v27 = v33;
        }
        *(_BYTE *)(v28 + v19 + 16) = 0;
        *(_QWORD *)v28 = v19;
        *(_BYTE *)(v28 + 8) = 0;
        *v27 = v28;
        ++*(_DWORD *)(a1 + 412);
        v29 = (__int64 *)(*(_QWORD *)(a1 + 400) + 8LL * (unsigned int)sub_C929D0(v35, v26));
        v24 = *v29;
        if ( *v29 != -8 )
          goto LABEL_25;
        do
        {
          do
          {
            v24 = v29[1];
            ++v29;
          }
          while ( v24 == -8 );
LABEL_25:
          ;
        }
        while ( !v24 );
        goto LABEL_18;
      }
      while ( !(unsigned __int8)sub_C89090(v31, pw_name, pw_passwd, 0, 0) )
      {
        v31 += 2;
        if ( v38 == v31 )
          goto LABEL_16;
      }
    }
    v12 = 0;
    goto LABEL_16;
  }
  v12 = *(unsigned __int8 *)(*(_QWORD *)v11 + 8LL);
LABEL_5:
  if ( src != &v43 )
    _libc_free((unsigned __int64)src);
  return v12;
}
