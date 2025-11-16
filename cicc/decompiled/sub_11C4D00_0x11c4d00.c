// Function: sub_11C4D00
// Address: 0x11c4d00
//
__int64 __fastcall sub_11C4D00(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // rax
  unsigned __int8 *v6; // rsi
  unsigned int v7; // r15d
  _QWORD *v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // r9
  __int64 **v15[2]; // [rsp+0h] [rbp-1F0h] BYREF
  __int64 v16; // [rsp+10h] [rbp-1E0h]
  __int64 v17; // [rsp+18h] [rbp-1D8h] BYREF
  unsigned int v18; // [rsp+20h] [rbp-1D0h]
  _QWORD v19[2]; // [rsp+D8h] [rbp-118h] BYREF
  _BYTE v20[192]; // [rsp+E8h] [rbp-108h] BYREF
  unsigned __int8 *v21; // [rsp+1A8h] [rbp-48h]
  __int64 v22; // [rsp+1B0h] [rbp-40h]
  __int64 v23; // [rsp+1B8h] [rbp-38h]

  v15[1] = 0;
  v16 = 1;
  v15[0] = (__int64 **)sub_B43CA0((__int64)a1);
  v5 = &v17;
  do
  {
    *v5 = -4096;
    v5 += 3;
    *((_DWORD *)v5 - 4) = 100;
  }
  while ( v5 != v19 );
  v6 = a1;
  v23 = a3;
  v19[1] = 0x800000000LL;
  v19[0] = v20;
  v21 = a1;
  v22 = a2;
  sub_11C48A0((__int64)v15, a1);
  v7 = 0;
  v8 = (_QWORD *)sub_11BF430(v15);
  v9 = (__int64)v8;
  if ( v8 )
  {
    v6 = a1 + 24;
    v7 = 1;
    sub_B44220(v8, (__int64)(a1 + 24), 0);
    if ( a2 )
    {
      v6 = (unsigned __int8 *)v9;
      sub_CFEAE0(a2, v9, v10, v11, v12, v13);
    }
  }
  if ( (_BYTE *)v19[0] != v20 )
    _libc_free(v19[0], v6);
  if ( (v16 & 1) == 0 )
    sub_C7D6A0(v17, 24LL * v18, 8);
  return v7;
}
