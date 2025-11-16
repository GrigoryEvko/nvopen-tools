// Function: sub_7DF750
// Address: 0x7df750
//
_QWORD *__fastcall sub_7DF750(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, char a5)
{
  _QWORD *v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  char v12; // r13
  __int64 v13; // rdx

  v8 = sub_725D60();
  *((_BYTE *)v8 + 144) |= 0x42u;
  v9 = v8;
  v8[1] = a1;
  v8[15] = a2;
  v8[16] = a3;
  sub_877E20(0, v8, a4);
  v10 = *(_QWORD *)(a4 + 160);
  if ( v10 )
  {
    v11 = 0;
    v12 = a5 & 1;
    while ( *(_QWORD *)(v10 + 128) <= a3 && (*(_QWORD *)(v10 + 128) != a3 || !v12 || (*(_BYTE *)(v10 + 146) & 8) == 0) )
    {
      v13 = *(_QWORD *)(v10 + 112);
      v11 = v10;
      if ( !v13 )
        goto LABEL_11;
      v10 = *(_QWORD *)(v10 + 112);
    }
    if ( !v11 )
      goto LABEL_13;
    v13 = v10;
    v10 = v11;
LABEL_11:
    *(_QWORD *)(v10 + 112) = v9;
  }
  else
  {
LABEL_13:
    *(_QWORD *)(a4 + 160) = v9;
    v13 = v10;
  }
  v9[14] = v13;
  return v9;
}
