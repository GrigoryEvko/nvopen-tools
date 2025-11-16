// Function: sub_16E87C0
// Address: 0x16e87c0
//
__int64 __fastcall sub_16E87C0(__int64 a1, __int64 *a2)
{
  char v2; // al
  __int64 v3; // rdx
  void *v4; // rax
  __int64 v5; // rsi
  size_t v7; // rdx
  unsigned int v8; // esi
  char *v9; // [rsp+0h] [rbp-70h] BYREF
  __int64 v10; // [rsp+8h] [rbp-68h]
  _BYTE v11[16]; // [rsp+10h] [rbp-60h] BYREF
  void *v12; // [rsp+20h] [rbp-50h] BYREF
  __int64 v13; // [rsp+28h] [rbp-48h]
  __int64 v14; // [rsp+30h] [rbp-40h]
  __int64 v15; // [rsp+38h] [rbp-38h]
  int v16; // [rsp+40h] [rbp-30h]
  char **v17; // [rsp+48h] [rbp-28h]

  if ( *((_BYTE *)a2 + 20) )
  {
    v2 = *((_BYTE *)a2 + 22);
    if ( *((_BYTE *)a2 + 21) )
      v3 = 2 * (unsigned int)(v2 != 0);
    else
      v3 = v2 == 0 ? 1 : 3;
    v4 = (void *)*((unsigned int *)a2 + 4);
    v5 = *a2;
    LOBYTE(v13) = 1;
    v12 = v4;
    sub_16F4F70(a1, v5, v3, &v12);
    return a1;
  }
  v17 = &v9;
  v10 = 0x1000000000LL;
  v12 = &unk_49EFC48;
  v9 = v11;
  v16 = 1;
  v15 = 0;
  v14 = 0;
  v13 = 0;
  sub_16E7A40((__int64)&v12, 0, 0, 0);
  sub_16F4F50(&v12, a2[1], 0, 0);
  v7 = (unsigned int)v10;
  v8 = *((_DWORD *)a2 + 4);
  if ( v8 > (unsigned int)v10 )
  {
    sub_16E8750(a1, v8 - v10);
    v7 = (unsigned int)v10;
  }
  sub_16E7EE0(a1, v9, v7);
  v12 = &unk_49EFD28;
  sub_16E7960((__int64)&v12);
  if ( v9 == v11 )
    return a1;
  _libc_free((unsigned __int64)v9);
  return a1;
}
