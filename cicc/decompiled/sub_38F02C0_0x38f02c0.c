// Function: sub_38F02C0
// Address: 0x38f02c0
//
__int64 __fastcall sub_38F02C0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  int v9; // edx
  __int64 v10; // r8
  __int64 (*v11)(); // rax
  char v12; // al
  unsigned int v13; // r8d
  char v15; // al
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int8 v19; // r8
  int v21; // eax
  unsigned __int8 v22; // [rsp+10h] [rbp-1C0h]
  __int64 v23; // [rsp+10h] [rbp-1C0h]
  __int64 v24; // [rsp+10h] [rbp-1C0h]
  unsigned __int8 v25; // [rsp+18h] [rbp-1B8h]
  _QWORD v26[2]; // [rsp+20h] [rbp-1B0h] BYREF
  __int64 v27; // [rsp+30h] [rbp-1A0h] BYREF
  unsigned __int64 v28; // [rsp+38h] [rbp-198h] BYREF
  const char *v29; // [rsp+40h] [rbp-190h] BYREF
  char v30; // [rsp+50h] [rbp-180h]
  char v31; // [rsp+51h] [rbp-17Fh]
  const char *v32; // [rsp+60h] [rbp-170h] BYREF
  _QWORD *v33; // [rsp+68h] [rbp-168h]
  __int64 v34; // [rsp+70h] [rbp-160h]
  __int64 v35; // [rsp+78h] [rbp-158h]
  int v36; // [rsp+80h] [rbp-150h]
  const char ***v37; // [rsp+88h] [rbp-148h]
  const char **v38; // [rsp+90h] [rbp-140h] BYREF
  __int64 v39; // [rsp+98h] [rbp-138h]
  _WORD v40[152]; // [rsp+A0h] [rbp-130h] BYREF

  v26[0] = a3;
  v26[1] = a4;
  v5 = sub_3909460(a1);
  v6 = sub_39092A0(v5);
  v38 = 0;
  v7 = v6;
  if ( sub_38EB6A0(a1, &v27, (__int64)&v38) )
    return 1;
  v8 = *(_QWORD *)(a1 + 328);
  v9 = 0;
  v10 = v27;
  v11 = *(__int64 (**)())(*(_QWORD *)v8 + 72LL);
  if ( v11 != sub_168DB40 )
  {
    v24 = v27;
    v21 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v11)(v8, &v27, 0);
    v10 = v24;
    v9 = v21;
  }
  v12 = sub_38CF2B0(v10, &v28, v9);
  if ( !v12 )
  {
    v32 = "unexpected token in '";
    v33 = v26;
    LOWORD(v34) = 1283;
    v38 = &v32;
    v40[0] = 770;
    v39 = (__int64)"' directive";
    return (unsigned int)sub_3909790(a1, v7, &v38, 0, 0);
  }
  v22 = v12;
  v31 = 1;
  v29 = "Count is negative";
  v30 = 3;
  v15 = sub_3909C80(a1, v28 >> 63, v7, &v29);
  v13 = v22;
  if ( !v15 )
  {
    LOWORD(v34) = 1283;
    v32 = "unexpected token in '";
    v33 = v26;
    v40[0] = 770;
    v38 = &v32;
    v39 = (__int64)"' directive";
    v16 = sub_3909E20(a1, 9, &v38);
    v13 = v22;
    if ( !v16 )
    {
      v23 = sub_38EFA80(a1, a2);
      if ( v23 )
      {
        v38 = (const char **)v40;
        v39 = 0x10000000000LL;
        v36 = 1;
        v35 = 0;
        v34 = 0;
        v33 = 0;
        v32 = (const char *)&unk_49EFC48;
        v37 = &v38;
        sub_16E7A40((__int64)&v32, 0, 0, 0);
        while ( v28-- )
        {
          v17 = sub_3909460(a1);
          v18 = sub_39092A0(v17);
          v19 = sub_38E48B0(
                  a1,
                  (__int64)&v32,
                  *(unsigned __int8 **)(v23 + 16),
                  *(_QWORD *)(v23 + 24),
                  0,
                  0,
                  0,
                  0,
                  0,
                  v18);
          if ( v19 )
            goto LABEL_14;
        }
        sub_38EF860(a1, a2, &v32);
        v19 = 0;
LABEL_14:
        v25 = v19;
        v32 = (const char *)&unk_49EFD28;
        sub_16E7960((__int64)&v32);
        v13 = v25;
        if ( v38 != (const char **)v40 )
        {
          _libc_free((unsigned __int64)v38);
          return v25;
        }
        return v13;
      }
      return 1;
    }
  }
  return v13;
}
