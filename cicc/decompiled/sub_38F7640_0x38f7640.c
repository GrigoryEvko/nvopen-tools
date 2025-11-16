// Function: sub_38F7640
// Address: 0x38f7640
//
__int64 __fastcall sub_38F7640(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int8 v4; // al
  unsigned int v5; // r12d
  unsigned __int64 *v6; // r14
  unsigned __int64 *v7; // r15
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  __int64 v11; // rbx
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rdi
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 *v21; // [rsp+8h] [rbp-208h]
  __int64 v22; // [rsp+10h] [rbp-200h]
  unsigned __int64 *v23; // [rsp+30h] [rbp-1E0h] BYREF
  unsigned __int64 *v24; // [rsp+38h] [rbp-1D8h]
  __int64 v25; // [rsp+40h] [rbp-1D0h]
  const char *v26; // [rsp+50h] [rbp-1C0h] BYREF
  char v27; // [rsp+60h] [rbp-1B0h]
  char v28; // [rsp+61h] [rbp-1AFh]
  __int64 v29[2]; // [rsp+70h] [rbp-1A0h] BYREF
  unsigned __int64 v30; // [rsp+80h] [rbp-190h]
  __int64 v31; // [rsp+88h] [rbp-188h]
  __int64 v32; // [rsp+90h] [rbp-180h]
  __int16 v33; // [rsp+98h] [rbp-178h]
  _QWORD v34[2]; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v35; // [rsp+B0h] [rbp-160h]
  __int64 v36; // [rsp+B8h] [rbp-158h]
  int v37; // [rsp+C0h] [rbp-150h]
  unsigned __int64 *v38; // [rsp+C8h] [rbp-148h]
  unsigned __int64 v39[2]; // [rsp+D0h] [rbp-140h] BYREF
  _BYTE v40[304]; // [rsp+E0h] [rbp-130h] BYREF

  v33 = 0;
  v29[0] = 0;
  v29[1] = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v28 = 1;
  v26 = "expected identifier in '.irp' directive";
  v27 = 3;
  v4 = sub_38F0EE0(a1, v29, a3, a4);
  if ( (unsigned __int8)sub_3909CB0(a1, v4, &v26)
    || (v34[0] = "expected comma in '.irp' directive", LOWORD(v35) = 259, (unsigned __int8)sub_3909E20(a1, 25, v34))
    || (unsigned __int8)sub_38F6810(a1, 0, (__int64 *)&v23, v15, v16, v17)
    || (v40[1] = 1,
        v39[0] = (unsigned __int64)"expected End of Statement",
        v40[0] = 3,
        (unsigned __int8)sub_3909E20(a1, 9, v39)) )
  {
    v5 = 1;
  }
  else
  {
    v5 = 1;
    v22 = sub_38EFA80(a1, a2);
    if ( v22 )
    {
      v39[0] = (unsigned __int64)v40;
      v39[1] = 0x10000000000LL;
      v38 = v39;
      v37 = 1;
      v34[0] = &unk_49EFC48;
      v36 = 0;
      v35 = 0;
      v34[1] = 0;
      sub_16E7A40((__int64)v34, 0, 0, 0);
      v18 = (__int64)v23;
      v21 = v24;
      if ( v24 == v23 )
      {
LABEL_35:
        v5 = 0;
        sub_38EF860(a1, a2, v34);
      }
      else
      {
        while ( 1 )
        {
          v19 = sub_3909460(a1);
          v20 = sub_39092A0(v19);
          v5 = sub_38E48B0(
                 a1,
                 (__int64)v34,
                 *(unsigned __int8 **)(v22 + 16),
                 *(_QWORD *)(v22 + 24),
                 (__int64)v29,
                 1,
                 v18,
                 1u,
                 1,
                 v20);
          if ( (_BYTE)v5 )
            break;
          v18 += 24;
          if ( v21 == (unsigned __int64 *)v18 )
            goto LABEL_35;
        }
      }
      v34[0] = &unk_49EFD28;
      sub_16E7960((__int64)v34);
      if ( (_BYTE *)v39[0] != v40 )
        _libc_free(v39[0]);
    }
  }
  v6 = v24;
  v7 = v23;
  if ( v24 != v23 )
  {
    do
    {
      v8 = v7[1];
      v9 = *v7;
      if ( v8 != *v7 )
      {
        do
        {
          if ( *(_DWORD *)(v9 + 32) > 0x40u )
          {
            v10 = *(_QWORD *)(v9 + 24);
            if ( v10 )
              j_j___libc_free_0_0(v10);
          }
          v9 += 40LL;
        }
        while ( v8 != v9 );
        v9 = *v7;
      }
      if ( v9 )
        j_j___libc_free_0(v9);
      v7 += 3;
    }
    while ( v6 != v7 );
    v7 = v23;
  }
  if ( v7 )
    j_j___libc_free_0((unsigned __int64)v7);
  v11 = v31;
  v12 = v30;
  if ( v31 != v30 )
  {
    do
    {
      if ( *(_DWORD *)(v12 + 32) > 0x40u )
      {
        v13 = *(_QWORD *)(v12 + 24);
        if ( v13 )
          j_j___libc_free_0_0(v13);
      }
      v12 += 40LL;
    }
    while ( v11 != v12 );
    v12 = v30;
  }
  if ( v12 )
    j_j___libc_free_0(v12);
  return v5;
}
