// Function: sub_C7E920
// Address: 0xc7e920
//
__int64 __fastcall sub_C7E920(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        char a5,
        unsigned __int8 a6,
        unsigned __int8 a7,
        unsigned __int16 a8)
{
  char v9; // dl
  char v10; // al
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v16; // rax
  int v17; // eax
  int v20; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v21; // [rsp+20h] [rbp-60h] BYREF
  char v22; // [rsp+28h] [rbp-58h]
  __int64 v23[2]; // [rsp+30h] [rbp-50h] BYREF
  char v24; // [rsp+40h] [rbp-40h]

  sub_C83520(&v21, a2, a5 != 0 ? 3 : 0, 0);
  v9 = v22 & 1;
  v10 = (2 * (v22 & 1)) | v22 & 0xFD;
  v22 = v10;
  if ( v9 )
  {
    v22 = v10 & 0xFD;
    v11 = v21;
    v21 = 0;
    v23[0] = v11 | 1;
    v12 = sub_C64300(v23, a2);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v12;
    v13 = v23[0];
    *(_QWORD *)(a1 + 8) = v14;
    if ( (v13 & 1) != 0 || (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(v23, (__int64)a2);
  }
  else
  {
    v20 = v21;
    sub_C7E470((__int64)v23, v21, (const char **)a2, (__int64 *)0xFFFFFFFFFFFFFFFFLL, a3, a4, a6, a7, a8);
    sub_C83820(&v20);
    if ( (v24 & 1) != 0 )
    {
      v17 = v23[0];
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = v17;
      *(_QWORD *)(a1 + 8) = v23[1];
    }
    else
    {
      v16 = v23[0];
      *(_BYTE *)(a1 + 16) &= ~1u;
      *(_QWORD *)a1 = v16;
    }
  }
  if ( (v22 & 2) != 0 )
    sub_C0EC50(&v21);
  if ( (v22 & 1) != 0 && v21 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v21 + 8LL))(v21);
  return a1;
}
