// Function: sub_1403A90
// Address: 0x1403a90
//
_QWORD *__fastcall sub_1403A90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *v5; // r12
  _BYTE *v6; // r14
  _BYTE *v7; // rdi
  size_t v8; // r13
  __int64 v10; // rax
  size_t v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_22077B0(200);
  v5 = (_QWORD *)v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = 0;
    v6 = *(_BYTE **)a3;
    v7 = (_BYTE *)(v4 + 184);
    *(_QWORD *)(v4 + 16) = &unk_4F992F2;
    v8 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v4 + 80) = v4 + 64;
    *(_QWORD *)(v4 + 88) = v4 + 64;
    *(_QWORD *)(v4 + 128) = v4 + 112;
    *(_QWORD *)(v4 + 136) = v4 + 112;
    *(_QWORD *)v4 = off_49EAD28;
    *(_DWORD *)(v4 + 24) = 2;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    *(_DWORD *)(v4 + 64) = 0;
    *(_QWORD *)(v4 + 72) = 0;
    *(_QWORD *)(v4 + 96) = 0;
    *(_DWORD *)(v4 + 112) = 0;
    *(_QWORD *)(v4 + 120) = 0;
    *(_QWORD *)(v4 + 144) = 0;
    *(_BYTE *)(v4 + 152) = 0;
    *(_QWORD *)(v4 + 160) = a2;
    *(_QWORD *)(v4 + 168) = v4 + 184;
    if ( &v6[v8] && !v6 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v11[0] = v8;
    if ( v8 > 0xF )
    {
      v10 = sub_22409D0(v4 + 168, v11, 0);
      v5[21] = v10;
      v7 = (_BYTE *)v10;
      v5[23] = v11[0];
    }
    else
    {
      if ( v8 == 1 )
      {
        *(_BYTE *)(v4 + 184) = *v6;
LABEL_7:
        v5[22] = v8;
        v7[v8] = 0;
        return v5;
      }
      if ( !v8 )
        goto LABEL_7;
    }
    memcpy(v7, v6, v8);
    v8 = v11[0];
    v7 = (_BYTE *)v5[21];
    goto LABEL_7;
  }
  return v5;
}
