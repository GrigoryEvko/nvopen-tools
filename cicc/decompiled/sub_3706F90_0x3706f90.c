// Function: sub_3706F90
// Address: 0x3706f90
//
void __fastcall sub_3706F90(_QWORD *a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  _WORD v4[2]; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v5; // [rsp+10h] [rbp-50h] BYREF
  __int64 v6; // [rsp+18h] [rbp-48h] BYREF
  __int64 v7; // [rsp+20h] [rbp-40h] BYREF
  __int64 v8; // [rsp+28h] [rbp-38h] BYREF
  _QWORD v9[6]; // [rsp+30h] [rbp-30h] BYREF

  v4[1] = 4611;
  v9[1] = 4;
  *a1 = &unk_4A3C7C8;
  v4[0] = 2;
  v9[0] = v4;
  sub_370CE40(&v5, a1 + 2, v9);
  v2 = v5;
  v5 = 0;
  v6 = 0;
  v8 = v2 | 1;
  sub_3706DA0(&v7, &v8);
  if ( (v7 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  if ( (v8 & 1) != 0 || (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v8, (__int64)&v8);
  if ( (v6 & 1) != 0 || (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v6, (__int64)&v8);
  if ( (v5 & 1) != 0 || (v5 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v5, (__int64)&v8);
  v3 = a1[4];
  if ( (_QWORD *)v3 != a1 + 6 )
    _libc_free(v3);
}
