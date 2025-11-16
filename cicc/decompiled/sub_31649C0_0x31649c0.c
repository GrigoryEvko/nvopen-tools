// Function: sub_31649C0
// Address: 0x31649c0
//
void __fastcall sub_31649C0(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int8 *v7; // r14
  unsigned __int8 *v8; // rbx
  unsigned __int8 *v9; // r13
  __int64 *v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned __int64 *v17; // r12
  char v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rsi
  __int64 v21; // r8
  __int64 v22; // rax
  unsigned __int8 v23; // dl
  unsigned __int8 **v24; // rax
  __int64 v25; // rax
  unsigned __int8 v26; // dl
  unsigned __int8 **v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rbx
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  int v35; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v36; // [rsp+8h] [rbp-98h]
  __int64 v37; // [rsp+18h] [rbp-88h] BYREF
  __int64 v38; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int8 *v39; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int8 *v40[2]; // [rsp+30h] [rbp-70h] BYREF
  char v41; // [rsp+40h] [rbp-60h]
  unsigned __int64 *v42; // [rsp+50h] [rbp-50h] BYREF
  __int64 v43; // [rsp+58h] [rbp-48h]
  char v44; // [rsp+60h] [rbp-40h]

  v5 = sub_B43CB0(a2);
  v6 = *(_QWORD *)(a2 - 32);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_55;
  v35 = *(_DWORD *)(v6 + 36);
  v7 = (unsigned __int8 *)sub_B58EB0(a2, 0);
  sub_3164240(
    (__int64)v40,
    a1,
    a3,
    v5,
    v7,
    *(_QWORD **)(*(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL),
    v35 != 71 && v35 != 68);
  if ( !v41 )
    return;
  v8 = v40[0];
  v9 = v40[1];
  sub_B59720(a2, (__int64)v7, v40[0]);
  v10 = (__int64 *)(*((_QWORD *)v9 + 1) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*((_QWORD *)v9 + 1) & 4) != 0 )
    v10 = (__int64 *)*v10;
  v11 = sub_B9F6F0(v10, v9);
  v12 = a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)v12 )
  {
    v13 = *(_QWORD *)(v12 + 8);
    **(_QWORD **)(v12 + 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(v12 + 16);
  }
  *(_QWORD *)v12 = v11;
  if ( v11 )
  {
    v14 = *(_QWORD *)(v11 + 16);
    *(_QWORD *)(v12 + 8) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = v12 + 8;
    *(_QWORD *)(v12 + 16) = v11 + 16;
    *(_QWORD *)(v11 + 16) = v12;
  }
  v15 = *(_QWORD *)(a2 - 32);
  if ( !v15 || *(_BYTE *)v15 || *(_QWORD *)(v15 + 24) != *(_QWORD *)(a2 + 80) )
LABEL_55:
    BUG();
  if ( *(_DWORD *)(v15 + 36) == 69 )
  {
    if ( *v8 <= 0x1Cu )
    {
      if ( *v8 != 22 )
        return;
      v30 = *(_QWORD *)(v5 + 80);
      if ( !v30 )
        BUG();
      v17 = *(unsigned __int64 **)(v30 + 32);
      LOWORD(v43) = 1;
      goto LABEL_35;
    }
    sub_B445D0((__int64)&v42, (char *)v8);
    v16 = *((_QWORD *)v8 + 6);
    v17 = v42;
    v18 = v44;
    v37 = v16;
    if ( v16 )
      sub_B96E90((__int64)&v37, v16, 1);
    v19 = *(_QWORD *)(a2 + 48);
    v38 = v19;
    if ( v19 )
    {
      sub_B96E90((__int64)&v38, v19, 1);
      v20 = v37;
      v21 = v38;
      if ( !v37 )
      {
LABEL_31:
        if ( v21 )
        {
          sub_B91220((__int64)&v38, v21);
          v20 = v37;
          goto LABEL_33;
        }
        v29 = v37;
LABEL_40:
        v20 = v29;
LABEL_33:
        if ( !v20 )
          goto LABEL_34;
        goto LABEL_38;
      }
      if ( v38 )
      {
        v22 = sub_B10CD0((__int64)&v38);
        v23 = *(_BYTE *)(v22 - 16);
        if ( (v23 & 2) != 0 )
          v24 = *(unsigned __int8 ***)(v22 - 32);
        else
          v24 = (unsigned __int8 **)(v22 - 16 - 8LL * ((v23 >> 2) & 0xF));
        v36 = sub_AF34D0(*v24);
        v25 = sub_B10CD0((__int64)&v37);
        v26 = *(_BYTE *)(v25 - 16);
        if ( (v26 & 2) != 0 )
          v27 = *(unsigned __int8 ***)(v25 - 32);
        else
          v27 = (unsigned __int8 **)(v25 - 16 - 8LL * ((v26 >> 2) & 0xF));
        if ( v36 != sub_AF34D0(*v27) )
          goto LABEL_30;
        v31 = *((_QWORD *)v8 + 6);
        v39 = (unsigned __int8 *)v31;
        if ( v31 )
        {
          v32 = a2 + 48;
          sub_B96E90((__int64)&v39, v31, 1);
          v33 = *(_QWORD *)(a2 + 48);
          if ( !v33 )
            goto LABEL_49;
        }
        else
        {
          v33 = *(_QWORD *)(a2 + 48);
          v32 = a2 + 48;
          if ( !v33 )
          {
LABEL_30:
            v21 = v38;
            goto LABEL_31;
          }
        }
        sub_B91220(v32, v33);
LABEL_49:
        v34 = v39;
        *(_QWORD *)(a2 + 48) = v39;
        if ( v34 )
          sub_B976B0((__int64)&v39, v34, v32);
        goto LABEL_30;
      }
    }
    else
    {
      v29 = v37;
      v20 = v37;
      if ( !v37 )
        goto LABEL_40;
    }
LABEL_38:
    sub_B91220((__int64)&v37, v20);
LABEL_34:
    if ( !v18 )
      return;
LABEL_35:
    if ( !v17 )
      BUG();
    v28 = v17[2];
    v42 = v17;
    sub_B44550((_QWORD *)a2, v28, v17, v43);
  }
}
