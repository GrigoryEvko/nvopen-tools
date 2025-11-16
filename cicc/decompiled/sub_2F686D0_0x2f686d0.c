// Function: sub_2F686D0
// Address: 0x2f686d0
//
void __fastcall sub_2F686D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 *v8; // rdx
  __int64 *v9; // rsi
  unsigned __int64 v10; // r12
  unsigned __int64 v11[2]; // [rsp+0h] [rbp-A0h] BYREF
  _BYTE v12[48]; // [rsp+10h] [rbp-90h] BYREF
  _BYTE *v13; // [rsp+40h] [rbp-60h]
  __int64 v14; // [rsp+48h] [rbp-58h]
  _BYTE v15[16]; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 v16; // [rsp+60h] [rbp-40h]

  v6 = *a1;
  v7 = *(_DWORD *)(a2 + 8);
  v8 = *(__int64 **)(*a1 + 8);
  v9 = *(__int64 **)(*a1 + 16);
  if ( v7 )
  {
    v16 = 0;
    v11[1] = 0x200000000LL;
    v14 = 0x200000000LL;
    v11[0] = (unsigned __int64)v12;
    v13 = v15;
    sub_2F68500((__int64)v11, v9, v8, a4, a5);
    sub_2F67010(*(_QWORD *)v6, a2, (__int64)v11, *(_QWORD *)(a2 + 112), *(_DWORD **)(a2 + 120), *(_DWORD **)(v6 + 24));
    v10 = v16;
    if ( v16 )
    {
      sub_2F61B70(*(_QWORD *)(v16 + 16));
      j_j___libc_free_0(v10);
    }
    if ( v13 != v15 )
      _libc_free((unsigned __int64)v13);
    if ( (_BYTE *)v11[0] != v12 )
      _libc_free(v11[0]);
  }
  else
  {
    sub_2F68500(a2, v9, v8, a4, a5);
  }
}
