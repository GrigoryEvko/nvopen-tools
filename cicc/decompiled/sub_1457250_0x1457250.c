// Function: sub_1457250
// Address: 0x1457250
//
__int64 __fastcall sub_1457250(__int64 a1)
{
  unsigned int v1; // r12d
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r14
  signed __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi

  v1 = *(unsigned __int8 *)(a1 + 48);
  if ( !(_BYTE)v1 )
    return v1;
  v3 = *(_QWORD **)a1;
  v4 = 24LL * *(unsigned int *)(a1 + 8);
  v5 = (_QWORD *)(*(_QWORD *)a1 + v4);
  v6 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 3);
  if ( v6 >> 2 )
  {
    v7 = &v3[12 * (v6 >> 2)];
    while ( 1 )
    {
      v8 = v3[2];
      if ( v8 )
      {
        if ( !sub_1452CB0(v8) )
          goto LABEL_7;
      }
      v9 = v3[5];
      if ( v9 && !sub_1452CB0(v9) )
      {
        LOBYTE(v1) = v5 == v3 + 3;
        return v1;
      }
      v10 = v3[8];
      if ( v10 && !sub_1452CB0(v10) )
      {
        LOBYTE(v1) = v5 == v3 + 6;
        return v1;
      }
      v11 = v3[11];
      if ( v11 && !sub_1452CB0(v11) )
      {
        LOBYTE(v1) = v5 == v3 + 9;
        return v1;
      }
      v3 += 12;
      if ( v7 == v3 )
      {
        v6 = 0xAAAAAAAAAAAAAAABLL * (v5 - v3);
        break;
      }
    }
  }
  if ( v6 == 2 )
  {
LABEL_28:
    v14 = v3[2];
    if ( v14 && !sub_1452CB0(v14) )
      goto LABEL_7;
    v3 += 3;
    goto LABEL_22;
  }
  if ( v6 == 3 )
  {
    v13 = v3[2];
    if ( v13 && !sub_1452CB0(v13) )
      goto LABEL_7;
    v3 += 3;
    goto LABEL_28;
  }
  if ( v6 != 1 )
    return v1;
LABEL_22:
  v12 = v3[2];
  if ( !v12 || sub_1452CB0(v12) )
    return v1;
LABEL_7:
  LOBYTE(v1) = v5 == v3;
  return v1;
}
