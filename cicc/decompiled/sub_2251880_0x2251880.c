// Function: sub_2251880
// Address: 0x2251880
//
unsigned __int64 __fastcall sub_2251880(const wchar_t **a1, size_t a2, __int64 a3, const wchar_t *a4, size_t a5)
{
  _QWORD *v5; // r13
  const wchar_t *v9; // rcx
  size_t v10; // r14
  unsigned __int64 v11; // rdx
  wchar_t *v12; // rax
  wchar_t *v13; // r15
  const wchar_t *v14; // rsi
  wchar_t *v15; // rdi
  unsigned __int64 v16; // rdi
  const wchar_t *v17; // rsi
  unsigned __int64 result; // rax
  size_t v20; // [rsp+8h] [rbp-50h]
  unsigned __int64 v21[8]; // [rsp+18h] [rbp-40h] BYREF

  v5 = a1 + 2;
  v9 = a1[1];
  v20 = a2 + a3;
  v10 = (size_t)v9 - a2 - a3;
  v21[0] = (unsigned __int64)v9 + a5 - a3;
  if ( a1 + 2 == (const wchar_t **)*a1 )
    v11 = 3;
  else
    v11 = (unsigned __int64)a1[2];
  v12 = (wchar_t *)sub_22517A0((__int64)a1, v21, v11);
  v13 = v12;
  if ( a2 )
  {
    v14 = *a1;
    if ( a2 == 1 )
      *v12 = *v14;
    else
      wmemcpy(v12, v14, a2);
  }
  if ( a4 && a5 )
  {
    v15 = &v13[a2];
    if ( a5 == 1 )
      *v15 = *a4;
    else
      wmemcpy(v15, a4, a5);
  }
  v16 = (unsigned __int64)*a1;
  if ( v10 )
  {
    v17 = (const wchar_t *)(v16 + 4 * v20);
    if ( v10 == 1 )
    {
      v13[a2 + a5] = *v17;
    }
    else
    {
      wmemcpy(&v13[a2 + a5], v17, v10);
      v16 = (unsigned __int64)*a1;
    }
  }
  if ( v5 != (_QWORD *)v16 )
    j___libc_free_0(v16);
  result = v21[0];
  *a1 = v13;
  a1[2] = (const wchar_t *)result;
  return result;
}
