// Function: sub_1C1AF40
// Address: 0x1c1af40
//
__int64 __fastcall sub_1C1AF40(__int64 a1, __int64 a2, __int64 *a3)
{
  char *v5; // rdi
  char v6; // al
  _BOOL8 v7; // rcx
  __int64 result; // rax
  __int64 v9; // rax
  char v10; // al
  _BOOL8 v11; // rcx
  __int64 v12; // rax
  void *v13; // rax
  size_t v14; // r12
  char v15; // [rsp+7h] [rbp-39h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h] BYREF
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
               &v15,
               &v16);
    if ( (_BYTE)result )
    {
      sub_1C15400(a1, &src);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v16);
      result = n;
    }
    else
    {
      if ( v15 )
      {
LABEL_15:
        *a3 = 0;
        return result;
      }
      result = n;
    }
    if ( result )
    {
      v12 = sub_16E4080(a1);
      v13 = (void *)sub_145CBF0(*(__int64 **)(v12 + 8), n + 1, 1);
      v14 = n;
      result = (__int64)memcpy(v13, src, n);
      *(_BYTE *)(result + v14) = 0;
      *a3 = result;
      return result;
    }
    goto LABEL_15;
  }
  v5 = (char *)*a3;
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
             &v15,
             &v16);
  if ( (_BYTE)result )
  {
    sub_1C15400(a1, &src);
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v16);
  }
  return result;
}
