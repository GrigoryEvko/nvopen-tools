// Function: sub_3112550
// Address: 0x3112550
//
__int64 sub_3112550()
{
  __int64 v0; // rax
  unsigned __int64 *v1; // r12
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // r13
  unsigned __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // rsi
  unsigned __int8 v7; // al
  unsigned __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // r12
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // r14
  unsigned __int64 v17; // rdi
  __int64 v18; // [rsp+0h] [rbp-80h] BYREF
  __int64 v19; // [rsp+8h] [rbp-78h] BYREF
  _QWORD *v20; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int8 v21; // [rsp+18h] [rbp-68h]
  __int64 v22[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v23; // [rsp+40h] [rbp-40h]

  v0 = sub_22077B0(0x18u);
  if ( v0 )
  {
    *(_QWORD *)(v0 + 16) = 0;
    *(_OWORD *)v0 = 0;
  }
  v1 = (unsigned __int64 *)unk_5031F48;
  unk_5031F48 = v0;
  if ( v1 )
  {
    v2 = v1[1];
    if ( v2 )
      sub_3111400(v2);
    v3 = *v1;
    if ( *v1 )
    {
      sub_3112140(v3 + 16);
      v4 = *(_QWORD *)(v3 + 16);
      if ( v4 != v3 + 64 )
        j_j___libc_free_0(v4);
      j_j___libc_free_0(v3);
    }
    j_j___libc_free_0((unsigned __int64)v1);
  }
  if ( (_BYTE)qword_50321C8 || (result = (__int64)&qword_5031F60, LOBYTE(qword_5031FA8[8])) )
  {
    result = unk_5031F48;
    *(_BYTE *)(unk_5031F48 + 16LL) = 1;
    return result;
  }
  if ( qword_50320D0 )
  {
    sub_CA41E0(&v18);
    v6 = (__int64)v22;
    v23 = 260;
    v22[0] = (__int64)&qword_50320C8;
    sub_3113320(&v20, v22, v18);
    v7 = v21;
    v21 &= ~2u;
    if ( (v7 & 1) != 0 )
    {
      v8 = (unsigned __int64)v20;
      v20 = 0;
      v19 = v8 | 1;
      if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v22[0] = v8 | 1;
        v6 = qword_50320C8;
        v19 = 0;
        sub_3112270(v22, qword_50320C8, qword_50320D0);
        if ( (v22[0] & 1) != 0 || (v22[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(v22, v6);
        if ( (v19 & 1) != 0 || (v19 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v19, v6);
LABEL_22:
        result = v21;
        if ( (v21 & 2) != 0 )
          sub_31124E0(&v20, v6);
        if ( v20 )
          result = (*(__int64 (__fastcall **)(_QWORD *))(*v20 + 8LL))(v20);
        v9 = v18;
        if ( v18 )
        {
          if ( !_InterlockedSub((volatile signed __int32 *)(v18 + 8), 1u) )
            return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
        }
        return result;
      }
      v10 = 0;
    }
    else
    {
      v10 = v20;
    }
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *))(*v10 + 40LL))(v10) )
    {
      v14 = unk_5031F48;
      v15 = v10[6];
      v10[6] = 0;
      v16 = *(_QWORD *)v14;
      *(_QWORD *)v14 = v15;
      if ( v16 )
      {
        sub_3112140(v16 + 16);
        v17 = *(_QWORD *)(v16 + 16);
        if ( v17 != v16 + 64 )
          j_j___libc_free_0(v17);
        v6 = 72;
        j_j___libc_free_0(v16);
      }
      *(_BYTE *)(v14 + 16) = 0;
    }
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD *))(*v10 + 48LL))(v10) )
    {
      v11 = unk_5031F48;
      v12 = v10[7];
      v10[7] = 0;
      v13 = *(_QWORD *)(v11 + 8);
      *(_QWORD *)(v11 + 8) = v12;
      if ( v13 )
        sub_3111400(v13);
      *(_BYTE *)(v11 + 16) = 0;
    }
    goto LABEL_22;
  }
  return result;
}
