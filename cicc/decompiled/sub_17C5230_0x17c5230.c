// Function: sub_17C5230
// Address: 0x17c5230
//
_BOOL8 __fastcall sub_17C5230(__int64 a1)
{
  __int64 v1; // rbx
  _QWORD *v2; // rdi
  _BOOL4 v3; // r12d
  __int64 v5; // [rsp+0h] [rbp-1B0h] BYREF
  __int16 v6; // [rsp+10h] [rbp-1A0h]
  __int64 v7; // [rsp+20h] [rbp-190h] BYREF
  __int16 v8; // [rsp+30h] [rbp-180h]
  __int64 v9; // [rsp+40h] [rbp-170h] BYREF
  __int16 v10; // [rsp+50h] [rbp-160h]
  __int64 v11; // [rsp+60h] [rbp-150h] BYREF
  __int16 v12; // [rsp+70h] [rbp-140h]
  __int64 *v13; // [rsp+80h] [rbp-130h] BYREF
  __int64 v14; // [rsp+90h] [rbp-120h] BYREF
  int v15; // [rsp+ACh] [rbp-104h]
  __int64 *v16; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v17; // [rsp+D0h] [rbp-E0h] BYREF
  int v18; // [rsp+ECh] [rbp-C4h]
  _QWORD v19[2]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+110h] [rbp-A0h] BYREF
  int v21; // [rsp+12Ch] [rbp-84h]
  _QWORD v22[2]; // [rsp+140h] [rbp-70h] BYREF
  _QWORD v23[2]; // [rsp+150h] [rbp-60h] BYREF
  int v24; // [rsp+160h] [rbp-50h]
  int v25; // [rsp+168h] [rbp-48h]
  unsigned int v26; // [rsp+16Ch] [rbp-44h]

  v1 = a1 + 240;
  LOWORD(v20) = 260;
  v19[0] = a1 + 240;
  sub_16E1010((__int64)v22, (__int64)v19);
  if ( v26 > 0x1E )
  {
    v2 = (_QWORD *)v22[0];
LABEL_3:
    if ( v2 != v23 )
      j_j___libc_free_0(v2, v23[0] + 1LL);
    v5 = v1;
    v3 = 0;
    v6 = 260;
    sub_16E1010((__int64)&v13, (__int64)&v5);
    if ( v15 != 9 )
    {
      v7 = v1;
      v8 = 260;
      sub_16E1010((__int64)&v16, (__int64)&v7);
      if ( v18 != 5 )
      {
        v9 = v1;
        v10 = 260;
        sub_16E1010((__int64)v19, (__int64)&v9);
        if ( v21 != 6 )
        {
          v11 = v1;
          v12 = 260;
          v3 = 1;
          sub_16E1010((__int64)v22, (__int64)&v11);
          if ( v24 == 32 && v25 == 3 )
            LOBYTE(v3) = v26 != 27;
          if ( (_QWORD *)v22[0] != v23 )
            j_j___libc_free_0(v22[0], v23[0] + 1LL);
        }
        if ( (__int64 *)v19[0] != &v20 )
          j_j___libc_free_0(v19[0], v20 + 1);
      }
      if ( v16 != &v17 )
        j_j___libc_free_0(v16, v17 + 1);
    }
    if ( v13 != &v14 )
      j_j___libc_free_0(v13, v14 + 1);
    return v3;
  }
  v2 = (_QWORD *)v22[0];
  v3 = ((0x60000888uLL >> v26) & 1) == 0;
  if ( ((0x60000888uLL >> v26) & 1) == 0 )
    goto LABEL_3;
  if ( (_QWORD *)v22[0] != v23 )
    j_j___libc_free_0(v22[0], v23[0] + 1LL);
  return v3;
}
