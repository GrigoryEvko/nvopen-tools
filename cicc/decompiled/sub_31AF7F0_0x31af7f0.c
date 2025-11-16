// Function: sub_31AF7F0
// Address: 0x31af7f0
//
char __fastcall sub_31AF7F0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rsi
  _QWORD *v3; // rdi
  _QWORD *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  bool v7; // al
  __int64 v8; // rdi
  bool v9; // r8
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v26; // [rsp-58h] [rbp-58h] BYREF
  __int64 v27; // [rsp-50h] [rbp-50h]
  __int64 v28; // [rsp-48h] [rbp-48h] BYREF
  __int64 v29; // [rsp-40h] [rbp-40h]

  v2 = *a2;
  v3 = (_QWORD *)*a1;
  if ( *(_DWORD *)(v3[6] + 72LL) == 2 )
  {
    LOBYTE(v11) = nullsub_2034(v3, v2);
    return v11;
  }
  v4 = v3;
  v5 = v3[4];
  if ( !v5 )
    goto LABEL_23;
  if ( v2 == v5 )
  {
    v6 = v3[5];
    if ( v6 == v2 )
      goto LABEL_8;
  }
  else
  {
    if ( !sub_B445A0(*(_QWORD *)(v5 + 16), *(_QWORD *)(v2 + 16)) )
    {
      v5 = v3[4];
      goto LABEL_23;
    }
    v6 = v3[5];
    if ( v2 == v6 )
    {
LABEL_25:
      v5 = v3[4];
      if ( !v5 )
        goto LABEL_28;
      if ( !v2 )
      {
LABEL_27:
        v10 = v4[5];
        goto LABEL_11;
      }
LABEL_8:
      v8 = *(_QWORD *)(v5 + 16);
      v5 = v2;
      if ( sub_B445A0(v8, *(_QWORD *)(v2 + 16)) )
        v5 = v4[4];
      v9 = sub_B445A0(*(_QWORD *)(v4[5] + 16LL), *(_QWORD *)(v2 + 16));
      v10 = v2;
      if ( v9 )
        goto LABEL_11;
      goto LABEL_27;
    }
  }
  v7 = sub_B445A0(*(_QWORD *)(v2 + 16), *(_QWORD *)(v6 + 16));
  v5 = v3[4];
  if ( !v7 )
  {
LABEL_23:
    if ( v5 != sub_318B4B0(v2) )
    {
      v15 = v3[5];
      v11 = sub_318B520(v2);
      if ( v15 != v11 )
        return v11;
    }
    goto LABEL_25;
  }
  if ( v5 )
    goto LABEL_8;
LABEL_28:
  v10 = v2;
  v5 = v2;
LABEL_11:
  v4[4] = v5;
  v4[5] = v10;
  v11 = sub_31BAED0(v4, v2);
  v12 = v11;
  if ( *(_DWORD *)(v11 + 16) == 1 )
  {
    v13 = sub_31B9B30(v4, v11, 0, 0);
    if ( v13 )
    {
      *(_QWORD *)(v13 + 48) = v12;
      *(_QWORD *)(v12 + 40) = v13;
    }
    v14 = sub_31B9BF0(v4, v12, 0, 0);
    if ( v14 )
    {
      *(_QWORD *)(v14 + 40) = v12;
      *(_QWORD *)(v12 + 48) = v14;
    }
    if ( sub_B445A0(*(_QWORD *)(v4[4] + 16LL), *(_QWORD *)(v2 + 16)) )
    {
      v21 = sub_318B520(v2);
      v22 = v4[4];
      v27 = v21;
      v26 = v22;
      v23 = sub_31B9080(&v26, v4);
      v29 = v24;
      v28 = v23;
      sub_31BB4E0(v4, v12, &v28);
    }
    LOBYTE(v11) = sub_B445A0(*(_QWORD *)(v2 + 16), *(_QWORD *)(v4[5] + 16LL));
    if ( (_BYTE)v11 )
    {
      v16 = v4[5];
      v17 = sub_318B4B0(v2);
      v27 = v16;
      v26 = v17;
      v11 = sub_31B9080(&v26, v4);
      v19 = v11;
      v20 = v18;
      if ( v18 )
        v20 = *(_QWORD *)(v18 + 48);
      if ( v20 != v11 )
      {
        do
        {
          v28 = v12;
          v29 = v12;
          LOBYTE(v11) = sub_31BB4E0(v4, v19, &v28);
          v19 = *(_QWORD *)(v19 + 48);
        }
        while ( v20 != v19 );
      }
    }
  }
  return v11;
}
