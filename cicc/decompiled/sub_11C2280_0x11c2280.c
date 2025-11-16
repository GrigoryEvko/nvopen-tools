// Function: sub_11C2280
// Address: 0x11c2280
//
__int64 __fastcall sub_11C2280(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 **v8; // rax
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  _QWORD *v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // r12
  __int64 v15; // r12
  unsigned __int8 *v17; // [rsp-10h] [rbp-210h]
  __int64 **v19[2]; // [rsp+10h] [rbp-1F0h] BYREF
  __int64 v20; // [rsp+20h] [rbp-1E0h]
  __int64 v21; // [rsp+28h] [rbp-1D8h] BYREF
  unsigned int v22; // [rsp+30h] [rbp-1D0h]
  _QWORD v23[2]; // [rsp+E8h] [rbp-118h] BYREF
  _BYTE v24[192]; // [rsp+F8h] [rbp-108h] BYREF
  __int64 v25; // [rsp+1B8h] [rbp-48h]
  __int64 v26; // [rsp+1C0h] [rbp-40h]
  __int64 v27; // [rsp+1C8h] [rbp-38h]

  v7 = a1;
  v8 = (__int64 **)sub_B43CA0(a3);
  v11 = a4;
  v19[1] = 0;
  v12 = v23;
  v20 = 1;
  v19[0] = v8;
  v13 = &v21;
  do
  {
    *v13 = -4096;
    v13 += 3;
    *((_DWORD *)v13 - 4) = 100;
  }
  while ( v13 != v23 );
  v25 = a3;
  v23[1] = 0x800000000LL;
  v14 = a1 + 24 * a2;
  v23[0] = v24;
  v26 = a4;
  v27 = a5;
  if ( a1 != v14 )
  {
    do
    {
      v17 = *(unsigned __int8 **)(v7 + 16);
      v7 += 24;
      sub_11C1FA0((__int64)v19, a2, (__int64)v12, v11, v9, v10, *(_OWORD *)(v7 - 24), v17);
    }
    while ( v14 != v7 );
  }
  v15 = sub_11BF430(v19);
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0], a2);
  if ( (v20 & 1) == 0 )
    sub_C7D6A0(v21, 24LL * v22, 8);
  return v15;
}
