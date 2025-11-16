// Function: sub_221FF50
// Address: 0x221ff50
//
__int64 __fastcall sub_221FF50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9)
{
  __int64 *v9; // rdi
  __int64 v10; // r13
  wchar_t *v12; // rdi
  size_t v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  wchar_t *v16; // rax
  int v17; // [rsp+1Ch] [rbp-84h] BYREF
  wchar_t *v18; // [rsp+20h] [rbp-80h] BYREF
  size_t n; // [rsp+28h] [rbp-78h]
  wchar_t s2[4]; // [rsp+30h] [rbp-70h] BYREF
  const wchar_t *v21[4]; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v22)(const wchar_t **); // [rsp+60h] [rbp-40h]

  v9 = *(__int64 **)(a1 + 16);
  v22 = 0;
  v17 = 0;
  v10 = sub_22144B0(v9, a2, a3, a4, a5, a6, a7, &v17, 0, (__int64)v21);
  if ( v17 )
  {
    *a8 = v17;
    goto LABEL_3;
  }
  if ( !v22 )
    sub_426248((__int64)"uninitialized __any_string");
  v18 = s2;
  sub_221FEA0((__int64 *)&v18, v21[0], (__int64)&v21[0][(__int64)v21[1]]);
  v12 = *(wchar_t **)a9;
  v13 = n;
  if ( v18 == s2 )
  {
    v16 = s2;
    if ( n )
    {
      if ( n == 1 )
      {
        *v12 = s2[0];
      }
      else
      {
        wmemcpy(v12, s2, n);
        v13 = n;
        v12 = *(wchar_t **)a9;
        v16 = v18;
      }
    }
    *(_QWORD *)(a9 + 8) = v13;
    v12[v13] = 0;
    v12 = v16;
  }
  else
  {
    v14 = *(_QWORD *)s2;
    if ( v12 == (wchar_t *)(a9 + 16) )
    {
      *(_QWORD *)a9 = v18;
      *(_QWORD *)(a9 + 8) = v13;
      *(_QWORD *)(a9 + 16) = v14;
      goto LABEL_3;
    }
    v15 = *(_QWORD *)(a9 + 16);
    *(_QWORD *)a9 = v18;
    *(_QWORD *)(a9 + 8) = v13;
    *(_QWORD *)(a9 + 16) = v14;
    if ( !v12 )
      goto LABEL_3;
    v18 = v12;
    *(_QWORD *)s2 = v15;
  }
  n = 0;
  *v12 = 0;
  if ( v12 != s2 )
    j___libc_free_0((unsigned __int64)v12);
LABEL_3:
  if ( v22 )
    v22(v21);
  return v10;
}
