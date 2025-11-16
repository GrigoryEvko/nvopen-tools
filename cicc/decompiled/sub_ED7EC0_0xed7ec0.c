// Function: sub_ED7EC0
// Address: 0xed7ec0
//
__int64 __fastcall sub_ED7EC0(__int64 a1, void **a2, __int64 *a3)
{
  __int64 v5; // rax
  _QWORD v6[2]; // [rsp+0h] [rbp-60h] BYREF
  char v7; // [rsp+10h] [rbp-50h]
  __int64 v8[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v9; // [rsp+30h] [rbp-30h] BYREF

  sub_CA0F50(v8, a2);
  if ( (unsigned int)sub_2241AC0(v8, "-") )
    sub_CA4130((__int64)v6, a3, (__int64)a2, -1, 1u, 0, 1);
  else
    sub_C7DF90((__int64)v6);
  if ( (__int64 *)v8[0] != &v9 )
    j_j___libc_free_0(v8[0], v9 + 1);
  if ( (v7 & 1) != 0 && LODWORD(v6[0]) )
  {
    sub_C63CA0(v8, v6[0], v6[1]);
    v5 = v8[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v5 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v7 & 1) == 0 && v6[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v6[0] + 8LL))(v6[0]);
  }
  else
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = v6[0];
  }
  return a1;
}
