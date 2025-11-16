// Function: sub_2240BB0
// Address: 0x2240bb0
//
unsigned __int64 __fastcall sub_2240BB0(unsigned __int64 *a1, size_t a2, __int64 a3, _BYTE *a4, size_t a5)
{
  unsigned __int64 *v6; // r12
  unsigned __int64 v9; // rcx
  size_t v10; // r13
  unsigned __int64 v11; // rdx
  _BYTE *v12; // rax
  _BYTE *v13; // r15
  _BYTE *v14; // rsi
  _BYTE *v15; // rdi
  _BYTE *v16; // rsi
  _BYTE *v17; // rdi
  unsigned __int64 result; // rax
  size_t v20; // [rsp+8h] [rbp-50h]
  unsigned __int64 v21[8]; // [rsp+18h] [rbp-40h] BYREF

  v6 = a1 + 2;
  v9 = a1[1];
  v20 = a2 + a3;
  v10 = v9 - (a2 + a3);
  v21[0] = v9 + a5 - a3;
  if ( a1 + 2 == (unsigned __int64 *)*a1 )
    v11 = 15;
  else
    v11 = a1[2];
  v12 = (_BYTE *)sub_22409D0((__int64)a1, v21, v11);
  v13 = v12;
  if ( a2 )
  {
    v14 = (_BYTE *)*a1;
    if ( a2 == 1 )
      *v12 = *v14;
    else
      memcpy(v12, v14, a2);
  }
  if ( a4 && a5 )
  {
    v15 = &v13[a2];
    if ( a5 == 1 )
      *v15 = *a4;
    else
      memcpy(v15, a4, a5);
  }
  if ( v10 )
  {
    v16 = (_BYTE *)(*a1 + v20);
    v17 = &v13[a2 + a5];
    if ( v10 == 1 )
      *v17 = *v16;
    else
      memcpy(v17, v16, v10);
  }
  if ( v6 != (unsigned __int64 *)*a1 )
    j___libc_free_0(*a1);
  result = v21[0];
  *a1 = (unsigned __int64)v13;
  a1[2] = result;
  return result;
}
