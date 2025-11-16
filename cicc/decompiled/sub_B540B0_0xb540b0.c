// Function: sub_B540B0
// Address: 0xb540b0
//
void __fastcall sub_B540B0(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  char *v4; // rdi
  __int64 v5; // rdi
  char *v6; // [rsp+0h] [rbp-60h] BYREF
  __int64 v7; // [rsp+8h] [rbp-58h]
  _BYTE v8[80]; // [rsp+10h] [rbp-50h] BYREF

  v2 = sub_BC89C0(*a1);
  if ( !v2 )
    return;
  v3 = v2;
  if ( (unsigned int)sub_BC8980(v2) != (*(_DWORD *)(*a1 + 4LL) & 0x7FFFFFFu) >> 1 )
    BUG();
  v6 = v8;
  v7 = 0x800000000LL;
  if ( !(unsigned __int8)sub_BC8BD0(v3, &v6) )
  {
    v4 = v6;
    if ( v6 == v8 )
      return;
    goto LABEL_5;
  }
  v5 = (__int64)(a1 + 1);
  if ( *((_BYTE *)a1 + 56) )
  {
    sub_B48480(v5, &v6);
    v4 = v6;
    if ( v6 == v8 )
      return;
LABEL_5:
    _libc_free(v4, &v6);
    return;
  }
  a1[2] = 0x800000000LL;
  a1[1] = a1 + 3;
  if ( (_DWORD)v7 )
    sub_B48480(v5, &v6);
  v4 = v6;
  *((_BYTE *)a1 + 56) = 1;
  if ( v4 != v8 )
    goto LABEL_5;
}
