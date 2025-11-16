// Function: sub_C856B0
// Address: 0xc856b0
//
unsigned __int64 __fastcall sub_C856B0(
        __int64 a1,
        int *a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        char a6,
        unsigned int a7)
{
  int v8; // r12d
  char v9; // bl
  _BYTE *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rax
  _BYTE *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  _BYTE *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rax
  int v26; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+8h] [rbp-98h]
  int v28; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+30h] [rbp-70h]
  unsigned int v31; // [rsp+30h] [rbp-70h]
  __int64 v32; // [rsp+38h] [rbp-68h]
  _QWORD v33[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v8 = a5;
  v9 = a4;
  v28 = 128;
  sub_2241E40(a1, a2, a3, a4, a5);
  while ( 1 )
  {
    sub_C85490(a1, a3, v9);
    if ( v8 == 1 )
      break;
    if ( v8 == 2 )
    {
      v20 = (_BYTE *)*a3;
      v34 = 257;
      if ( *v20 )
      {
        v33[0] = v20;
        LOBYTE(v34) = 3;
      }
      v21 = sub_C825C0((__int64)v33, 0);
      v27 = v22;
      v32 = v21;
      v31 = v21;
      v25 = sub_2241E50(v33, 0, v22, v23, v24);
      if ( (_DWORD)v32 == 2 && v25 == v27 )
        return 0;
      if ( v31 )
        return v32;
    }
    else
    {
      v10 = (_BYTE *)*a3;
      v34 = 257;
      if ( *v10 )
      {
        v33[0] = v10;
        LOBYTE(v34) = 3;
      }
      v32 = sub_C82340((__int64)v33, 0, 0x1F8u);
      v30 = v11;
      if ( !(_DWORD)v32 )
        return 0;
      v14 = sub_2241E50(v33, 0, v11, v12, v13);
      if ( (_DWORD)v32 != 17 || v14 != v30 )
        return v32;
      v31 = 17;
    }
LABEL_20:
    if ( !--v28 )
      return v31 | v32 & 0xFFFFFFFF00000000LL;
  }
  v16 = (_BYTE *)*a3;
  v34 = 257;
  if ( *v16 )
  {
    v33[0] = v16;
    LOBYTE(v34) = 3;
  }
  v18 = sub_C83360((__int64)v33, a2, 1, 3, a6, a7);
  v31 = v18;
  v26 = v18;
  if ( (_DWORD)v18 )
  {
    if ( sub_2241E50(v33, a2, v17, v18, v19) != v17 || v26 != 17 && v26 != 13 )
      return v31 | v32 & 0xFFFFFFFF00000000LL;
    goto LABEL_20;
  }
  return 0;
}
