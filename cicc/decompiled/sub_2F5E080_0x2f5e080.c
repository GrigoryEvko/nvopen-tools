// Function: sub_2F5E080
// Address: 0x2f5e080
//
__int64 __fastcall sub_2F5E080(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rax
  unsigned int v5; // r12d
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rdi
  char v9; // al
  char v10; // al
  const char *v11; // rax
  __int64 v12; // [rsp+8h] [rbp-178h]
  const char *v13; // [rsp+10h] [rbp-170h] BYREF
  char v14; // [rsp+30h] [rbp-150h]
  char v15; // [rsp+31h] [rbp-14Fh]
  unsigned __int64 v16[2]; // [rsp+40h] [rbp-140h] BYREF
  _BYTE v17[72]; // [rsp+50h] [rbp-130h] BYREF
  int v18; // [rsp+98h] [rbp-E8h] BYREF
  unsigned __int64 v19; // [rsp+A0h] [rbp-E0h]
  int *v20; // [rsp+A8h] [rbp-D8h]
  int *v21; // [rsp+B0h] [rbp-D0h]
  __int64 v22; // [rsp+B8h] [rbp-C8h]
  unsigned __int64 v23[2]; // [rsp+C0h] [rbp-C0h] BYREF
  _BYTE v24[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v4 = *(__int64 **)(a1 + 768);
  *(_BYTE *)(a1 + 984) = 0;
  v12 = sub_B2BE50(*v4);
  v16[0] = (unsigned __int64)v17;
  v16[1] = 0x1000000000LL;
  v20 = &v18;
  v21 = &v18;
  v18 = 0;
  v19 = 0;
  v22 = 0;
  v23[0] = (unsigned __int64)v24;
  v23[1] = 0x800000000LL;
  v5 = sub_2F5D4A0(a1, a2, a3, (__int64)v16, (__int64)v23, 0);
  if ( v5 == -1 )
  {
    v9 = *(_BYTE *)(a1 + 984);
    if ( v9 )
    {
      v10 = v9 & 3;
      switch ( v10 )
      {
        case 1:
          v15 = 1;
          v11 = "register allocation failed: maximum depth for recoloring reached. Use -fexhaustive-register-search to skip cutoffs";
          break;
        case 2:
          v15 = 1;
          v11 = "register allocation failed: maximum interference for recoloring reached. Use -fexhaustive-register-searc"
                "h to skip cutoffs";
          break;
        case 3:
          v15 = 1;
          v11 = "register allocation failed: maximum interference and depth for recoloring reached. Use -fexhaustive-regi"
                "ster-search to skip cutoffs";
          break;
        default:
          goto LABEL_2;
      }
      v13 = v11;
      v14 = 3;
      sub_B6ECE0(v12, (__int64)&v13);
    }
  }
LABEL_2:
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0]);
  v6 = v19;
  while ( v6 )
  {
    sub_2F4E180(*(_QWORD *)(v6 + 24));
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
    j_j___libc_free_0(v7);
  }
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0]);
  return v5;
}
