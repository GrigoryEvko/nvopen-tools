// Function: sub_C2DD40
// Address: 0xc2dd40
//
unsigned __int64 *__fastcall sub_C2DD40(unsigned __int64 *a1, _BYTE *a2, size_t a3, __int64 a4, __int64 a5)
{
  size_t v5; // r15
  __int64 v9; // rax
  __int64 v10; // rbx
  const void *v11; // r9
  _BYTE *v12; // rdi
  __int64 v14; // rax
  size_t v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a3;
  v9 = sub_22077B0(80);
  v10 = v9;
  if ( v9 )
  {
    v11 = a2;
    v12 = (_BYTE *)(v9 + 24);
    *(_QWORD *)(v9 + 8) = v9 + 24;
    *(_QWORD *)v9 = &unk_49DBDA0;
    if ( &a2[a3] && !a2 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v16[0] = a3;
    if ( a3 > 0xF )
    {
      v14 = sub_22409D0(v9 + 8, v16, 0);
      v11 = a2;
      *(_QWORD *)(v10 + 8) = v14;
      v12 = (_BYTE *)v14;
      *(_QWORD *)(v10 + 24) = v16[0];
    }
    else
    {
      if ( a3 == 1 )
      {
        *(_BYTE *)(v9 + 24) = *a2;
LABEL_7:
        *(_QWORD *)(v10 + 16) = v5;
        v12[v5] = 0;
        *(_QWORD *)(v10 + 40) = a4;
        sub_CA0F50(v10 + 48, a5);
        goto LABEL_8;
      }
      if ( !a3 )
        goto LABEL_7;
    }
    memcpy(v12, v11, a3);
    v5 = v16[0];
    v12 = *(_BYTE **)(v10 + 8);
    goto LABEL_7;
  }
LABEL_8:
  *a1 = v10 | 1;
  return a1;
}
