// Function: sub_1682BF0
// Address: 0x1682bf0
//
int __fastcall sub_1682BF0(__int64 a1)
{
  char *v2; // rax
  char *v3; // r12
  size_t v4; // rax
  __int64 v5; // rax
  __int64 v6; // r12
  int result; // eax
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rcx
  int v16; // eax
  int v17; // eax
  int v18; // eax
  int v19; // eax
  unsigned __int64 v20; // r12
  __int64 v21; // rax
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rcx
  size_t v25; // rcx
  char *v26; // rdi
  size_t v27; // rdx
  __int64 v28; // rsi
  char *v29; // rdi
  char *v30; // rdi
  __int64 v31; // rsi
  size_t v32; // rdx
  char *v33; // [rsp+18h] [rbp-98h]
  _QWORD *v34; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v35; // [rsp+28h] [rbp-88h]
  _QWORD v36[2]; // [rsp+30h] [rbp-80h] BYREF
  char *file; // [rsp+40h] [rbp-70h] BYREF
  size_t v38; // [rsp+48h] [rbp-68h]
  _QWORD v39[2]; // [rsp+50h] [rbp-60h] BYREF
  char *v40; // [rsp+60h] [rbp-50h] BYREF
  size_t n; // [rsp+68h] [rbp-48h]
  _QWORD v42[8]; // [rsp+70h] [rbp-40h] BYREF

  v2 = getenv("MAKEFLAGS");
  if ( !v2 )
    return _InterlockedCompareExchange((volatile signed __int32 *)a1, 5, 0);
  v3 = v2;
  v34 = v36;
  v4 = strlen(v2);
  sub_1682310((__int64 *)&v34, v3, (__int64)&v3[v4]);
  v5 = sub_2241820(&v34, "--jobserver-auth=", -1, 17);
  v6 = v5;
  if ( v5 == -1 )
  {
    result = _InterlockedCompareExchange((volatile signed __int32 *)a1, 6, 0);
    goto LABEL_4;
  }
  v8 = v5 + 17;
  if ( v5 + 17 == sub_22416F0(&v34, "fifo:", v5 + 17, 5) )
  {
    v20 = v6 + 22;
    v21 = sub_22416F0(&v34, " ", v20, 1);
    v38 = 0;
    v22 = v21;
    LOBYTE(v39[0]) = 0;
    v33 = (char *)v39;
    file = (char *)v39;
    if ( v20 != -1 )
      goto LABEL_27;
    if ( v35 != -1 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v40 = (char *)v42;
    sub_1682310((__int64 *)&v40, (_BYTE *)v34 - 1, (__int64)v34 - 1);
    v30 = file;
    if ( v40 == (char *)v42 )
    {
      v32 = n;
      if ( n )
      {
        if ( n == 1 )
          *file = v42[0];
        else
          memcpy(file, v42, n);
        v32 = n;
        v30 = file;
      }
      v38 = v32;
      v30[v32] = 0;
      v30 = v40;
      goto LABEL_58;
    }
    if ( file == v33 )
    {
      file = v40;
      v38 = n;
      v39[0] = v42[0];
    }
    else
    {
      v31 = v39[0];
      file = v40;
      v38 = n;
      v39[0] = v42[0];
      if ( v30 )
      {
        v40 = v30;
        v42[0] = v31;
        goto LABEL_58;
      }
    }
    v40 = (char *)v42;
    v30 = (char *)v42;
LABEL_58:
    n = 0;
    *v30 = 0;
    if ( v40 != (char *)v42 )
      j_j___libc_free_0(v40, v42[0] + 1LL);
LABEL_27:
    if ( v20 > v22 )
    {
      result = _InterlockedCompareExchange((volatile signed __int32 *)a1, 7, 0);
      v29 = file;
      if ( file != v33 )
      {
LABEL_40:
        result = j_j___libc_free_0(v29, v39[0] + 1LL);
        goto LABEL_4;
      }
      goto LABEL_4;
    }
    v23 = v22 - v20;
    if ( v20 > v35 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v24 = v35 - v20;
    v40 = (char *)v42;
    if ( v35 - v20 > v23 )
      v24 = v23;
    sub_1682310((__int64 *)&v40, (_BYTE *)v34 + v20, (__int64)v34 + v20 + v24);
    v26 = file;
    if ( v40 == (char *)v42 )
    {
      v27 = n;
      if ( n )
      {
        if ( n == 1 )
          *file = v42[0];
        else
          memcpy(file, v42, n);
        v27 = n;
        v26 = file;
      }
      v38 = v27;
      v26[v27] = 0;
      v26 = v40;
      goto LABEL_35;
    }
    v27 = v42[0];
    v25 = n;
    if ( file == v33 )
    {
      file = v40;
      v38 = n;
      v39[0] = v42[0];
    }
    else
    {
      v28 = v39[0];
      file = v40;
      v38 = n;
      v39[0] = v42[0];
      if ( v26 )
      {
        v40 = v26;
        v42[0] = v28;
LABEL_35:
        n = 0;
        *v26 = 0;
        if ( v40 != (char *)v42 )
          j_j___libc_free_0(v40, v42[0] + 1LL);
        result = open(file, 2050, v27, v25);
        *(_DWORD *)(a1 + 188) = result;
        if ( result == -1 )
        {
          result = _InterlockedCompareExchange((volatile signed __int32 *)a1, 7, 0);
        }
        else
        {
          *(_DWORD *)(a1 + 192) = result;
          *(_BYTE *)(a1 + 204) = 1;
        }
LABEL_39:
        v29 = file;
        if ( file == v33 )
          goto LABEL_4;
        goto LABEL_40;
      }
    }
    v40 = (char *)v42;
    v26 = (char *)v42;
    goto LABEL_35;
  }
  v9 = sub_22416F0(&v34, ",", v8, 1);
  v10 = v9;
  if ( v9 == -1 || v8 > v9 )
  {
    result = _InterlockedCompareExchange((volatile signed __int32 *)a1, 7, 0);
  }
  else
  {
    v11 = v9 - v8;
    if ( v8 > v35 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v12 = v35 - v8;
    v33 = (char *)v39;
    file = (char *)v39;
    if ( v35 - v8 > v11 )
      v12 = v11;
    sub_1682310((__int64 *)&file, (_BYTE *)v34 + v8, (__int64)v34 + v8 + v12);
    v13 = sub_22416F0(&v34, " ", v10, 1);
    v14 = v10 + 1;
    if ( v13 != -1 )
      v13 += ~v10;
    if ( v14 > v35 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", (char)"basic_string::substr");
    v15 = v35 - v14;
    v40 = (char *)v42;
    if ( v35 - v14 > v13 )
      v15 = v13;
    sub_1682310((__int64 *)&v40, (_BYTE *)v34 + v14, (__int64)v34 + v14 + v15);
    if ( !v38 || !n || sub_22419D0(&file, "0123456789", 0, 10) != -1 || sub_22419D0(&v40, "0123456789", 0, 10) != -1 )
    {
      result = _InterlockedCompareExchange((volatile signed __int32 *)a1, 7, 0);
LABEL_42:
      if ( v40 != (char *)v42 )
        result = j_j___libc_free_0(v40, v42[0] + 1LL);
      goto LABEL_39;
    }
    v16 = sub_1682B50(
            (__int64 (__fastcall *)(__int64, _QWORD *, _QWORD))&strtol,
            (__int64)"stoi",
            (__int64)file,
            0,
            0xAu);
    *(_DWORD *)(a1 + 188) = v16;
    v17 = dup(v16);
    *(_DWORD *)(a1 + 188) = v17;
    if ( fcntl(v17, 2, 2048) != -1 )
    {
      v18 = sub_1682B50(
              (__int64 (__fastcall *)(__int64, _QWORD *, _QWORD))&strtol,
              (__int64)"stoi",
              (__int64)v40,
              0,
              0xAu);
      *(_DWORD *)(a1 + 192) = v18;
      v19 = dup(v18);
      *(_DWORD *)(a1 + 192) = v19;
      result = fcntl(v19, 2, 2048) + 1;
      if ( result )
      {
        *(_BYTE *)(a1 + 204) = 1;
        goto LABEL_42;
      }
      close(*(_DWORD *)(a1 + 188));
    }
    _InterlockedCompareExchange((volatile signed __int32 *)a1, 7, 0);
    sub_2240A30(&v40);
    result = sub_2240A30(&file);
  }
LABEL_4:
  if ( v34 != v36 )
    return j_j___libc_free_0(v34, v36[0] + 1LL);
  return result;
}
