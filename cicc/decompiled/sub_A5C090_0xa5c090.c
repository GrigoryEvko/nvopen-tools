// Function: sub_A5C090
// Address: 0xa5c090
//
_BYTE *__fastcall sub_A5C090(__int64 a1, __int64 a2, __int64 *a3)
{
  char v4; // al
  _BYTE *result; // rax
  __int64 *v7; // rbx
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rdi
  int v12; // eax
  __int64 (__fastcall *v13)(__int64, __int64); // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // [rsp+8h] [rbp-48h]
  int v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  char v23[8]; // [rsp+10h] [rbp-40h] BYREF
  char *v24; // [rsp+18h] [rbp-38h]

  v4 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 7 )
    return (_BYTE *)sub_A52D40(a1, a2);
  if ( v4 == 4 )
  {
    sub_904010(a1, "!DIArgList(");
    v7 = *(__int64 **)(a2 + 136);
    v23[0] = 1;
    v24 = ", ";
    v20 = &v7[*(unsigned int *)(a2 + 144)];
    while ( v20 != v7 )
    {
      v8 = *v7++;
      sub_A50EC0(a1, (__int64)v23);
      sub_A5C090(a1, v8);
    }
    return (_BYTE *)sub_904010(a1, ")");
  }
  else if ( (unsigned __int8)(v4 - 5) > 0x1Fu )
  {
    if ( v4 )
    {
      sub_A57EC0(a3[1], *(_QWORD *)(*(_QWORD *)(a2 + 136) + 8LL), a1);
      sub_A51310(a1, 0x20u);
      return sub_A5A730(a1, *(_QWORD *)(a2 + 136), (__int64)a3);
    }
    else
    {
      sub_904010(a1, "!\"");
      v14 = sub_B91420(a2, "!\"");
      sub_C92400(v14, v15, a1);
      return (_BYTE *)sub_A51310(a1, 0x22u);
    }
  }
  else
  {
    v9 = a3[2];
    v10 = 0;
    v11 = v9;
    if ( !v9 )
    {
      v22 = a3[3];
      v19 = sub_22077B0(400);
      v10 = v19;
      if ( v19 )
        sub_A55A10(v19, v22, 0);
      a3[2] = v10;
      v11 = v10;
    }
    v12 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 32LL))(v11, a2);
    if ( v12 == -1 )
    {
      if ( *(_BYTE *)a2 == 6 )
      {
        result = (_BYTE *)sub_A5CD70(a1, a2, a3);
      }
      else
      {
        v16 = sub_904010(a1, "<");
        v17 = sub_CB5A80(v16, a2);
        a2 = (__int64)">";
        result = (_BYTE *)sub_904010(v17, ">");
      }
    }
    else
    {
      v21 = v12;
      v18 = sub_A51310(a1, 0x21u);
      a2 = v21;
      result = (_BYTE *)sub_CB59F0(v18, v21);
    }
    a3[2] = v9;
    if ( v10 )
    {
      v13 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 8LL);
      if ( v13 == sub_A554F0 )
      {
        sub_A552A0(v10, a2);
        return (_BYTE *)j_j___libc_free_0(v10, 400);
      }
      else
      {
        return (_BYTE *)((__int64 (__fastcall *)(__int64))v13)(v10);
      }
    }
  }
  return result;
}
