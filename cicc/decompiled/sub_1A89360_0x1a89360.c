// Function: sub_1A89360
// Address: 0x1a89360
//
_QWORD *__fastcall sub_1A89360(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rax
  _BOOL4 v7; // r8d
  _QWORD *v8; // rbx
  __int64 v9; // rax
  __int64 v11; // rax
  _BOOL4 v12; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    v3 = a1 + 1;
    if ( v2 == (_QWORD *)a1[3] )
    {
      v7 = 1;
LABEL_11:
      v12 = v7;
      v8 = (_QWORD *)sub_22077B0(56);
      v8[4] = v8 + 4;
      v9 = *(_QWORD *)(a2 + 16);
      v8[5] = 1;
      v8[6] = v9;
      sub_220F040(v12, v8, v3, v2);
      ++a1[5];
      return v8;
    }
    v4 = *(_QWORD *)(a2 + 16);
LABEL_13:
    v11 = sub_220EF80(v3);
    if ( v4 <= *(_QWORD *)(v11 + 48) )
      return (_QWORD *)v11;
LABEL_9:
    v7 = 1;
    if ( v2 != v3 )
      v7 = v4 < v3[6];
    goto LABEL_11;
  }
  v4 = *(_QWORD *)(a2 + 16);
  while ( 1 )
  {
    v5 = v3[6];
    v6 = (_QWORD *)v3[3];
    if ( v4 < v5 )
      v6 = (_QWORD *)v3[2];
    if ( !v6 )
      break;
    v3 = v6;
  }
  if ( v4 < v5 )
  {
    if ( v3 == (_QWORD *)a1[3] )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v4 > v5 )
    goto LABEL_9;
  return v3;
}
