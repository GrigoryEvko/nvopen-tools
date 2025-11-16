// Function: sub_14570B0
// Address: 0x14570b0
//
unsigned __int64 __fastcall sub_14570B0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r15
  signed __int64 v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rdi
  unsigned __int64 result; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi

  v3 = *(_QWORD **)a1;
  v4 = 24LL * *(unsigned int *)(a1 + 8);
  v5 = (_QWORD *)(*(_QWORD *)a1 + v4);
  v6 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 3);
  if ( !(v6 >> 2) )
  {
LABEL_19:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          goto LABEL_6;
        goto LABEL_22;
      }
      v14 = v3[2];
      if ( v14 && !sub_1452CB0(v14) )
        goto LABEL_5;
      v3 += 3;
    }
    v15 = v3[2];
    if ( v15 && !sub_1452CB0(v15) )
      goto LABEL_5;
    v3 += 3;
LABEL_22:
    v13 = v3[2];
    if ( !v13 || sub_1452CB0(v13) )
      goto LABEL_6;
    goto LABEL_5;
  }
  v7 = &v3[12 * (v6 >> 2)];
  while ( 1 )
  {
    v8 = v3[2];
    if ( v8 )
    {
      if ( !sub_1452CB0(v8) )
        break;
    }
    v10 = v3[5];
    if ( v10 && !sub_1452CB0(v10) )
    {
      v3 += 3;
      break;
    }
    v11 = v3[8];
    if ( v11 && !sub_1452CB0(v11) )
    {
      v3 += 6;
      break;
    }
    v12 = v3[11];
    if ( v12 && !sub_1452CB0(v12) )
    {
      v3 += 9;
      break;
    }
    v3 += 12;
    if ( v7 == v3 )
    {
      v6 = 0xAAAAAAAAAAAAAAABLL * (v5 - v3);
      goto LABEL_19;
    }
  }
LABEL_5:
  if ( v5 != v3 )
    return sub_1456E90(a2);
LABEL_6:
  result = *(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !result )
    return sub_1456E90(a2);
  return result;
}
