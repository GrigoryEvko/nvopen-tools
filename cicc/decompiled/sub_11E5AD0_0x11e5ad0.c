// Function: sub_11E5AD0
// Address: 0x11e5ad0
//
unsigned __int64 __fastcall sub_11E5AD0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 **v11; // r15
  unsigned __int64 v12; // r13
  unsigned int v13; // ebx
  unsigned int v14; // eax
  __int64 v16; // [rsp+0h] [rbp-80h] BYREF
  __int64 v17; // [rsp+8h] [rbp-78h]
  _QWORD v18[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v19[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v20; // [rsp+40h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4);
  v5 = *(__int64 **)(a3 + 72);
  BYTE4(v17) = 0;
  v6 = *(_QWORD *)(a2 - 32LL * (v4 & 0x7FFFFFF));
  v7 = *(_QWORD *)(v6 + 8);
  v19[0] = "ctlz";
  v20 = 259;
  v18[0] = v6;
  v16 = v7;
  v18[1] = sub_ACD720(v5);
  v8 = sub_B33D10(a3, 0x41u, (__int64)&v16, 1, (int)v18, 2, v17, (__int64)v19);
  v20 = 257;
  v9 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v8 + 8), *(_DWORD *)(v7 + 8) >> 8, 0);
  v10 = sub_929DE0((unsigned int **)a3, v9, (_BYTE *)v8, (__int64)v19, 0, 0);
  v11 = *(__int64 ***)(a2 + 8);
  v20 = 257;
  v12 = v10;
  v13 = sub_BCB060(*(_QWORD *)(v10 + 8));
  v14 = sub_BCB060((__int64)v11);
  return sub_11DB4B0((__int64 *)a3, (unsigned int)(v13 <= v14) + 38, v12, v11, (__int64)v19, 0, v18[0], 0);
}
