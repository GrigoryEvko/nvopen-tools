// Function: sub_1ECAC00
// Address: 0x1ecac00
//
__int64 __fastcall sub_1ECAC00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rax
  __int64 v5; // r14
  unsigned int v6; // r12d
  __int64 v7; // rbx
  __int64 v8; // rdi
  char v10; // al
  char v11; // al
  const char *v12; // rax
  const char *v13; // [rsp+0h] [rbp-D0h] BYREF
  char v14; // [rsp+10h] [rbp-C0h]
  char v15; // [rsp+11h] [rbp-BFh]
  unsigned __int64 v16[2]; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE v17[72]; // [rsp+30h] [rbp-A0h] BYREF
  int v18; // [rsp+78h] [rbp-58h] BYREF
  __int64 v19; // [rsp+80h] [rbp-50h]
  int *v20; // [rsp+88h] [rbp-48h]
  int *v21; // [rsp+90h] [rbp-40h]
  __int64 v22; // [rsp+98h] [rbp-38h]

  v4 = *(__int64 **)(a1 + 680);
  *(_BYTE *)(a1 + 916) = 0;
  v5 = sub_15E0530(*v4);
  v16[1] = 0x1000000000LL;
  v16[0] = (unsigned __int64)v17;
  v18 = 0;
  v19 = 0;
  v20 = &v18;
  v21 = &v18;
  v22 = 0;
  v6 = sub_1EC7760(a1, a2, a3, (__int64)v16, 0);
  if ( v6 == -1 )
  {
    v10 = *(_BYTE *)(a1 + 916);
    if ( v10 )
    {
      v11 = v10 & 3;
      switch ( v11 )
      {
        case 1:
          v15 = 1;
          v12 = "register allocation failed: maximum depth for recoloring reached. Use -fexhaustive-register-search to skip cutoffs";
          break;
        case 2:
          v15 = 1;
          v12 = "register allocation failed: maximum interference for recoloring reached. Use -fexhaustive-register-searc"
                "h to skip cutoffs";
          break;
        case 3:
          v15 = 1;
          v12 = "register allocation failed: maximum interference and depth for recoloring reached. Use -fexhaustive-regi"
                "ster-search to skip cutoffs";
          break;
        default:
          goto LABEL_2;
      }
      v13 = v12;
      v14 = 3;
      sub_1602AC0(v5, (__int64)&v13);
    }
  }
LABEL_2:
  v7 = v19;
  while ( v7 )
  {
    sub_1EBC7A0(*(_QWORD *)(v7 + 24));
    v8 = v7;
    v7 = *(_QWORD *)(v7 + 16);
    j_j___libc_free_0(v8, 40);
  }
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0]);
  return v6;
}
