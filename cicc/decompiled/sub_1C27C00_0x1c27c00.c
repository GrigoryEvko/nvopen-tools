// Function: sub_1C27C00
// Address: 0x1c27c00
//
_QWORD *__fastcall sub_1C27C00(unsigned __int64 *a1, __int64 *a2)
{
  char v2; // al
  _BYTE *v3; // r14
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  _QWORD *result; // rax
  __int64 v11; // [rsp+8h] [rbp-88h] BYREF
  __int64 v12; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v13; // [rsp+18h] [rbp-78h] BYREF
  __int64 v14; // [rsp+20h] [rbp-70h] BYREF
  __int64 v15; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v16[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v17; // [rsp+40h] [rbp-50h]
  _QWORD *v18; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v19[6]; // [rsp+60h] [rbp-30h] BYREF

  v2 = *((_BYTE *)a2 + 16);
  if ( v2 )
  {
    if ( v2 == 1 )
    {
      v16[0] = "DataLayoutError: ";
      v17 = 259;
    }
    else
    {
      if ( *((_BYTE *)a2 + 17) == 1 )
        a2 = (__int64 *)*a2;
      else
        v2 = 2;
      v16[1] = a2;
      v16[0] = "DataLayoutError: ";
      LOBYTE(v17) = 3;
      HIBYTE(v17) = v2;
    }
  }
  else
  {
    v17 = 256;
  }
  sub_16E2FC0((__int64 *)&v18, (__int64)v16);
  v3 = v18;
  v4 = sub_16BCA90();
  sub_16BCCC0(&v12, v4, v5, v3);
  v6 = v12;
  v7 = *a1;
  *a1 = 0;
  v12 = 0;
  v15 = v6 | 1;
  v14 = v7 | 1;
  v11 = 0;
  sub_12BEC00(&v13, (unsigned __int64 *)&v14, (unsigned __int64 *)&v15);
  if ( (v14 & 1) != 0 || (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v14, (__int64)&v14, v8);
  if ( (v15 & 1) != 0 || (v15 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v15, (__int64)&v14, v8);
  v9 = *a1;
  if ( (*a1 & 1) != 0 || (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(a1, (__int64)&v14, v8);
  *a1 = v13 | v9 | 1;
  if ( (v11 & 1) != 0 || (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v11, (__int64)&v14, v8);
  if ( (v12 & 1) != 0 || (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v12, (__int64)&v14, v8);
  result = v19;
  if ( v18 != v19 )
    return (_QWORD *)j_j___libc_free_0(v18, v19[0] + 1LL);
  return result;
}
