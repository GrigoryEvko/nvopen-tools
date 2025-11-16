// Function: sub_36D8B00
// Address: 0x36d8b00
//
__int64 __fastcall sub_36D8B00(__int64 a1, float a2)
{
  _DWORD *v2; // r14
  _DWORD *v3; // rax
  void *v4; // r15
  unsigned int v5; // ecx
  __int64 v6; // rdx
  __int128 v7; // rax
  __int64 v8; // r9
  __int64 v9; // r13
  _QWORD *i; // rbx
  __int128 v11; // rax
  __int64 v12; // r9
  bool v14; // [rsp+1Fh] [rbp-81h] BYREF
  unsigned __int64 v15; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-78h]
  void *v17; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v18; // [rsp+38h] [rbp-68h]
  unsigned __int64 v19; // [rsp+50h] [rbp-50h] BYREF
  int v20; // [rsp+58h] [rbp-48h]

  v2 = sub_C33310();
  sub_C3B170((__int64)&v19, (__m128i)LODWORD(a2));
  sub_C407B0(&v17, (__int64 *)&v19, v2);
  sub_C338F0((__int64)&v19);
  v3 = sub_C33300();
  sub_C41640((__int64 *)&v17, v3, 1, &v14);
  v4 = sub_C33340();
  if ( **(_BYTE **)a1 )
  {
    if ( v17 == v4 )
      sub_C3E660((__int64)&v15, (__int64)&v17);
    else
      sub_C3A850((__int64)&v15, (__int64 *)&v17);
    v5 = v16;
    if ( 2 * v16 > 0x40 )
    {
      sub_C44A70((__int64)&v19, (__int64)&v15, (__int64)&v15);
      v5 = v16;
    }
    else
    {
      v20 = 2 * v16;
      v19 = v15 | (v15 << v16);
    }
    if ( v5 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    v6 = *(_QWORD *)(a1 + 16);
    v15 = v19;
    v16 = v20;
    *(_QWORD *)&v7 = sub_34007B0(
                       *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
                       (__int64)&v15,
                       v6,
                       7u,
                       0,
                       1u,
                       (__m128i)LODWORD(a2),
                       0);
    v9 = sub_33F7740(
           *(_QWORD **)(*(_QWORD *)(a1 + 8) + 64LL),
           1603,
           *(_QWORD *)(a1 + 16),
           **(_DWORD **)(a1 + 24),
           *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
           v8,
           v7);
    if ( v16 > 0x40 && v15 )
    {
      j_j___libc_free_0_0(v15);
      if ( v4 == v17 )
        goto LABEL_12;
LABEL_18:
      sub_C338F0((__int64)&v17);
      return v9;
    }
  }
  else
  {
    *(_QWORD *)&v11 = sub_33FE6E0(
                        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 64LL),
                        (__int64 *)&v17,
                        *(_QWORD *)(a1 + 16),
                        **(_QWORD **)(a1 + 24),
                        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
                        1,
                        (__m128i)LODWORD(a2));
    v9 = sub_33F7740(
           *(_QWORD **)(*(_QWORD *)(a1 + 8) + 64LL),
           373,
           *(_QWORD *)(a1 + 16),
           **(_DWORD **)(a1 + 24),
           *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
           v12,
           v11);
  }
  if ( v4 != v17 )
    goto LABEL_18;
LABEL_12:
  if ( v18 )
  {
    for ( i = &v18[3 * *(v18 - 1)]; v18 != i; sub_91D830(i) )
      i -= 3;
    j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
  }
  return v9;
}
