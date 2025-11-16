// Function: sub_17C5780
// Address: 0x17c5780
//
_QWORD *__fastcall sub_17C5780(__int16 *a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  _BYTE *v4; // r14
  _BYTE *v5; // rdi
  __int16 v6; // ax
  size_t v7; // r13
  __int64 v9; // rax
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = (_QWORD *)sub_22077B0(416);
  v3 = v2;
  if ( v2 )
  {
    v2[1] = 0;
    v4 = (_BYTE *)*((_QWORD *)a1 + 1);
    v5 = v2 + 23;
    v2[2] = &unk_4FA32AC;
    v2[10] = v2 + 8;
    v2[11] = v2 + 8;
    v2[16] = v2 + 14;
    v2[17] = v2 + 14;
    *v2 = off_49F0318;
    v6 = *a1;
    v7 = *((_QWORD *)a1 + 2);
    *((_DWORD *)v3 + 6) = 5;
    *((_WORD *)v3 + 80) = v6;
    v3[4] = 0;
    v3[5] = 0;
    v3[6] = 0;
    *((_DWORD *)v3 + 16) = 0;
    v3[9] = 0;
    v3[12] = 0;
    *((_DWORD *)v3 + 28) = 0;
    v3[15] = 0;
    v3[18] = 0;
    *((_BYTE *)v3 + 152) = 0;
    v3[21] = v3 + 23;
    if ( &v4[v7] && !v4 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v10[0] = v7;
    if ( v7 > 0xF )
    {
      v9 = sub_22409D0(v3 + 21, v10, 0);
      v3[21] = v9;
      v5 = (_BYTE *)v9;
      v3[23] = v10[0];
    }
    else
    {
      if ( v7 == 1 )
      {
        *((_BYTE *)v3 + 184) = *v4;
LABEL_7:
        v3[22] = v7;
        v5[v7] = 0;
        v3[26] = v3 + 28;
        v3[27] = 0;
        *((_BYTE *)v3 + 224) = 0;
        v3[30] = 0;
        v3[31] = 0;
        v3[32] = 0;
        v3[34] = 0;
        v3[35] = 0;
        v3[36] = 0;
        *((_DWORD *)v3 + 74) = 0;
        v3[38] = 0;
        v3[39] = 0;
        v3[40] = 0;
        v3[41] = 0;
        v3[42] = 0;
        v3[43] = 0;
        v3[46] = 0;
        v3[47] = 0;
        v3[48] = 0;
        v3[51] = 0;
        return v3;
      }
      if ( !v7 )
        goto LABEL_7;
    }
    memcpy(v5, v4, v7);
    v7 = v10[0];
    v5 = (_BYTE *)v3[21];
    goto LABEL_7;
  }
  return v3;
}
