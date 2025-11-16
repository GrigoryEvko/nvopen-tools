// Function: sub_38925C0
// Address: 0x38925c0
//
__int64 __fastcall sub_38925C0(__int64 a1, unsigned __int64 a2, const void *a3, size_t a4, __int64 *a5, __int64 *a6)
{
  __int64 v10; // r8
  int v11; // eax
  __int64 v12; // rdi
  unsigned int v13; // r13d
  __int64 v15; // rax
  int v16; // eax
  char v17; // r15
  unsigned int v18; // eax
  __int64 v19; // r8
  char v20; // al
  __int64 v21; // rax
  __int64 v23; // [rsp+18h] [rbp-88h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  char *v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+28h] [rbp-78h]
  char v27; // [rsp+30h] [rbp-70h] BYREF
  char v28; // [rsp+31h] [rbp-6Fh]

  v10 = *a5;
  if ( v10 && !a5[1] )
  {
    v28 = 1;
    v25 = "redefinition of type";
    v27 = 3;
    return (unsigned int)sub_38814C0(a1 + 8, a2, (__int64)&v25);
  }
  v11 = *(_DWORD *)(a1 + 64);
  if ( v11 == 200 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    v15 = *a5;
    a5[1] = 0;
    if ( !v15 )
    {
      v15 = sub_1644060(*(_QWORD *)a1, a3, a4);
      *a5 = v15;
    }
    *a6 = v15;
    return 0;
  }
  if ( v11 != 10 )
  {
    if ( v11 != 8 )
    {
      v12 = a1 + 8;
      if ( !v10 )
      {
        *a6 = 0;
        v28 = 1;
        v25 = "expected type";
        v27 = 3;
        return (unsigned int)sub_3891B00(a1, a6, (__int64)&v25, 0);
      }
LABEL_7:
      v28 = 1;
      v25 = "forward references to non-struct type";
      v27 = 3;
      return (unsigned int)sub_38814C0(v12, a2, (__int64)&v25);
    }
    v17 = 0;
LABEL_16:
    a5[1] = 0;
    if ( !v10 )
    {
      v21 = sub_1644060(*(_QWORD *)a1, a3, a4);
      *a5 = v21;
      v10 = v21;
    }
    v23 = v10;
    v25 = &v27;
    v26 = 0x800000000LL;
    v18 = sub_3892130(a1, (__int64)&v25);
    v19 = v23;
    v13 = v18;
    if ( (_BYTE)v18 || v17 && (v20 = sub_388AF10(a1, 11, "expected '>' in packed struct"), v19 = v23, v20) )
    {
      v13 = 1;
    }
    else
    {
      v24 = v19;
      sub_1643FB0(v19, v25, (unsigned int)v26, v17);
      *a6 = v24;
    }
    if ( v25 != &v27 )
      _libc_free((unsigned __int64)v25);
    return v13;
  }
  v16 = sub_3887100(a1 + 8);
  v12 = a1 + 8;
  *(_DWORD *)(a1 + 64) = v16;
  if ( v16 == 8 )
  {
    v10 = *a5;
    v17 = 1;
    goto LABEL_16;
  }
  if ( *a5 )
    goto LABEL_7;
  *a6 = 0;
  return sub_38923B0(a1, (__int64 **)a6, 1);
}
