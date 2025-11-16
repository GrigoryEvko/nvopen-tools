// Function: sub_15EA100
// Address: 0x15ea100
//
_QWORD *__fastcall sub_15EA100(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r12
  _BYTE *v5; // r14
  _BYTE *v6; // rdi
  size_t v7; // r13
  __int64 v9; // rax
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_22077B0(200);
  v4 = (_QWORD *)v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    v5 = *(_BYTE **)a2;
    v6 = (_BYTE *)(v3 + 184);
    *(_QWORD *)(v3 + 16) = &unk_4F9E22C;
    v7 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(v3 + 80) = v3 + 64;
    *(_QWORD *)(v3 + 88) = v3 + 64;
    *(_QWORD *)(v3 + 128) = v3 + 112;
    *(_QWORD *)(v3 + 136) = v3 + 112;
    *(_QWORD *)v3 = off_49ED328;
    *(_DWORD *)(v3 + 24) = 0;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_DWORD *)(v3 + 64) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_DWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 144) = 0;
    *(_BYTE *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = a1;
    *(_QWORD *)(v3 + 168) = v3 + 184;
    if ( &v5[v7] && !v5 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v10[0] = v7;
    if ( v7 > 0xF )
    {
      v9 = sub_22409D0(v3 + 168, v10, 0);
      v4[21] = v9;
      v6 = (_BYTE *)v9;
      v4[23] = v10[0];
    }
    else
    {
      if ( v7 == 1 )
      {
        *(_BYTE *)(v3 + 184) = *v5;
LABEL_7:
        v4[22] = v7;
        v6[v7] = 0;
        return v4;
      }
      if ( !v7 )
        goto LABEL_7;
    }
    memcpy(v6, v5, v7);
    v7 = v10[0];
    v6 = (_BYTE *)v4[21];
    goto LABEL_7;
  }
  return v4;
}
