// Function: sub_CD5150
// Address: 0xcd5150
//
size_t __fastcall sub_CD5150(__int64 a1, __int64 a2, char **a3)
{
  char *v5; // rdi
  char v6; // al
  _BOOL8 v7; // rcx
  size_t result; // rax
  __int64 v9; // rax
  char v10; // al
  _BOOL8 v11; // rcx
  char **v12; // rdi
  char *v13; // rcx
  __int64 v14; // rsi
  char *v15; // rcx
  char v16; // [rsp+7h] [rbp-39h] BYREF
  __int64 v17; // [rsp+8h] [rbp-38h] BYREF
  void *src; // [rsp+10h] [rbp-30h] BYREF
  size_t n; // [rsp+18h] [rbp-28h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v9 = *(_QWORD *)a1;
    src = 0;
    n = 0;
    v10 = (*(__int64 (__fastcall **)(__int64))(v9 + 16))(a1);
    v11 = 0;
    if ( v10 )
      v11 = n == 0;
    result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
               a1,
               a2,
               0,
               v11,
               &v16,
               &v17);
    if ( (_BYTE)result )
    {
      sub_CCDDA0(a1, &src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v17);
      result = n;
    }
    else
    {
      if ( v16 )
        goto LABEL_18;
      result = n;
    }
    if ( result )
    {
      v12 = *(char ***)(sub_CB0A70(a1) + 8);
      v13 = *v12;
      v14 = n + 1;
      v12[10] += n + 1;
      if ( v12[1] >= &v13[v14] && v13 )
        *v12 = &v13[v14];
      else
        v13 = (char *)sub_9D1E70((__int64)v12, v14, v14, 0);
      v15 = (char *)memcpy(v13, src, n);
      result = n;
      v15[n] = 0;
      *a3 = v15;
      return result;
    }
LABEL_18:
    *a3 = 0;
    return result;
  }
  v5 = *a3;
  if ( !*a3 )
    v5 = (char *)byte_3F871B3;
  src = v5;
  n = strlen(v5);
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = n == 0;
  result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             a2,
             0,
             v7,
             &v16,
             &v17);
  if ( (_BYTE)result )
  {
    sub_CCDDA0(a1, &src);
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v17);
  }
  return result;
}
