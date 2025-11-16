// Function: sub_20FC0D0
// Address: 0x20fc0d0
//
void __fastcall sub_20FC0D0(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // [rsp+0h] [rbp-90h] BYREF
  __int64 v12; // [rsp+8h] [rbp-88h]
  __int64 v13; // [rsp+10h] [rbp-80h]
  int v14; // [rsp+18h] [rbp-78h]
  unsigned __int64 v15[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v16[96]; // [rsp+30h] [rbp-60h] BYREF

  sub_20FAD80((__int64)a1);
  v2 = sub_1626D20(*a2);
  if ( *(_DWORD *)(*(_QWORD *)(v2 + 8 * (5LL - *(unsigned int *)(v2 + 8))) + 36LL) )
  {
    *a1 = a2;
    v15[0] = (unsigned __int64)v16;
    v15[1] = 0x400000000LL;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    sub_20FBA30(a1, (__int64)v15, (__int64)&v11);
    v7 = (__int64)a1[28];
    if ( v7 )
    {
      sub_20FA880((__int64)a1, v7, v3, v4, v5, v6);
      sub_20FA980((__int64)a1, (__int64)v15, (__int64)&v11, v8, v9, v10);
    }
    j___libc_free_0(v12);
    if ( (_BYTE *)v15[0] != v16 )
      _libc_free(v15[0]);
  }
}
