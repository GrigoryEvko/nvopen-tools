// Function: sub_AE5270
// Address: 0xae5270
//
char __fastcall sub_AE5270(__int64 a1, __int64 a2)
{
  __int16 v3; // dx
  char v4; // al
  unsigned __int8 v5; // bl
  unsigned __int16 v6; // ax
  __int64 v7; // r14
  __int64 v8; // r14
  _QWORD *v9; // rcx
  _BYTE *v10; // r9
  size_t v11; // r15
  _QWORD *v12; // rax
  size_t v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // rax
  _QWORD *v16; // rdi
  _BYTE *src; // [rsp+10h] [rbp-90h]
  _QWORD *v19; // [rsp+18h] [rbp-88h]
  _QWORD *v20; // [rsp+18h] [rbp-88h]
  size_t v21; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v22; // [rsp+30h] [rbp-70h] BYREF
  size_t v23; // [rsp+38h] [rbp-68h]
  _QWORD v24[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+50h] [rbp-50h]
  __int64 v26; // [rsp+58h] [rbp-48h]
  __int64 v27; // [rsp+60h] [rbp-40h]

  v3 = *(_WORD *)(a2 + 34) >> 1;
  v4 = v3 & 0x3F;
  if ( (v3 & 0x3F) == 0 || (v5 = v4 - 1, HIBYTE(v6) = 1, LOBYTE(v6) = v4 - 1, !HIBYTE(v6)) )
  {
    v8 = *(_QWORD *)(a2 + 24);
    v5 = sub_AE5260(a1, v8);
    if ( (unsigned __int8)sub_B2FC80(a2) == 1 || v5 > 3u )
      goto LABEL_6;
    v9 = *(_QWORD **)(a2 + 40);
    v22 = v24;
    v10 = (_BYTE *)v9[29];
    v11 = v9[30];
    if ( &v10[v11] && !v10 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v21 = v9[30];
    if ( v11 > 0xF )
    {
      src = v10;
      v19 = v9;
      v15 = sub_22409D0(&v22, &v21, 0);
      v9 = v19;
      v10 = src;
      v22 = (_QWORD *)v15;
      v16 = (_QWORD *)v15;
      v24[0] = v21;
    }
    else
    {
      if ( v11 == 1 )
      {
        LOBYTE(v24[0]) = *v10;
        v12 = v24;
        goto LABEL_16;
      }
      if ( !v11 )
      {
        v12 = v24;
LABEL_16:
        v23 = v11;
        *((_BYTE *)v12 + v11) = 0;
        v25 = v9[33];
        v26 = v9[34];
        v27 = v9[35];
        if ( (unsigned int)(v25 - 42) > 1 || *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8 != 3 )
        {
          if ( v22 != v24 )
            j_j___libc_free_0(v22, v24[0] + 1LL);
          v22 = (_QWORD *)sub_9208B0(a1, v8);
          v23 = v13;
          if ( (unsigned __int64)sub_CA1930(&v22) > 0x80 )
            v5 = 4;
          LOBYTE(v6) = v5;
          return v6;
        }
        if ( v22 != v24 )
        {
          j_j___libc_free_0(v22, v24[0] + 1LL);
          LOBYTE(v6) = v5;
          return v6;
        }
LABEL_6:
        LOBYTE(v6) = v5;
        return v6;
      }
      v16 = v24;
    }
    v20 = v9;
    memcpy(v16, v10, v11);
    v11 = v21;
    v12 = v22;
    v9 = v20;
    goto LABEL_16;
  }
  if ( (v3 & 0x200) == 0 )
  {
    v7 = *(_QWORD *)(a2 + 24);
    if ( v5 < (unsigned __int8)sub_AE5260(a1, v7) )
    {
      v14 = sub_AE5020(a1, v7);
      if ( v5 < v14 )
        v5 = v14;
    }
    sub_B2FC80(a2);
    goto LABEL_6;
  }
  return v6;
}
