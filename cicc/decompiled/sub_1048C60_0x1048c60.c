// Function: sub_1048C60
// Address: 0x1048c60
//
__int64 __fastcall sub_1048C60(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v4; // r13
  __int64 *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r11
  __int64 v19; // rax
  __int64 v20; // r10
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v2 = 0;
  v4 = (__int64 *)sub_B2BE50(a2);
  if ( (unsigned __int8)sub_B6E900((__int64)v4) )
  {
    v9 = *(__int64 **)(a1 + 8);
    v10 = *v9;
    v11 = v9[1];
    if ( v10 == v11 )
LABEL_27:
      BUG();
    while ( *(_UNKNOWN **)v10 != &unk_4F8EE48 )
    {
      v10 += 16;
      if ( v11 == v10 )
        goto LABEL_27;
    }
    v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(
            *(_QWORD *)(v10 + 8),
            &unk_4F8EE48);
    v13 = v12;
    v2 = (__int64 *)(v12 + 176);
    if ( !*(_BYTE *)(v12 + 184) )
    {
      v18 = *(_QWORD *)(v12 + 208);
      v19 = *(_QWORD *)(v12 + 200);
      v20 = *(_QWORD *)(v19 + 176);
      if ( !*(_BYTE *)(v20 + 280) )
      {
        v21 = v18;
        v22 = *(_QWORD *)(v19 + 176);
        sub_FF9360((_QWORD *)v22, *(_QWORD *)(v20 + 288), *(_QWORD *)(v20 + 296), *(__int64 **)(v20 + 304), 0, 0);
        v20 = v22;
        v18 = v21;
        *(_BYTE *)(v22 + 280) = 1;
      }
      sub_FE7D70(v2, *(const char **)(v13 + 192), v20, v18);
      *(_BYTE *)(v13 + 184) = 1;
    }
    if ( (unsigned __int8)sub_B6E980((__int64)v4) )
    {
      v14 = *(__int64 **)(a1 + 8);
      v15 = *v14;
      v16 = v14[1];
      if ( v15 == v16 )
LABEL_26:
        BUG();
      while ( *(_UNKNOWN **)v15 != &unk_4F87C64 )
      {
        v15 += 16;
        if ( v16 == v15 )
          goto LABEL_26;
      }
      v17 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v15 + 8) + 104LL))(
                          *(_QWORD *)(v15 + 8),
                          &unk_4F87C64)
                      + 176);
      if ( v17 )
      {
        LOBYTE(v24) = 1;
        v23 = sub_D844E0(v17);
        sub_B6E910(v4, v23, v24);
      }
    }
  }
  v5 = (__int64 *)sub_22077B0(24);
  if ( v5 )
  {
    *v5 = a2;
    v5[1] = (__int64)v2;
    v5[2] = 0;
  }
  v6 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v5;
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 16);
    if ( v7 )
    {
      sub_FDC110(*(__int64 **)(v6 + 16));
      j_j___libc_free_0(v7, 8);
    }
    j_j___libc_free_0(v6, 24);
  }
  return 0;
}
