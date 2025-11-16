// Function: sub_30A3E40
// Address: 0x30a3e40
//
_QWORD *__fastcall sub_30A3E40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  _QWORD *v5; // r12
  _BYTE *v6; // r14
  _BYTE *v7; // rdi
  size_t v8; // r13
  __int64 v10; // rax
  size_t v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_22077B0(0xD8u);
  v5 = (_QWORD *)v4;
  if ( v4 )
  {
    *(_QWORD *)(v4 + 8) = 0;
    v6 = *(_BYTE **)a3;
    v7 = (_BYTE *)(v4 + 192);
    *(_QWORD *)(v4 + 16) = &unk_502E0CD;
    v8 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(v4 + 56) = v4 + 104;
    *(_QWORD *)(v4 + 112) = v4 + 160;
    *(_QWORD *)v4 = off_4A31F08;
    *(_DWORD *)(v4 + 88) = 1065353216;
    *(_DWORD *)(v4 + 24) = 3;
    *(_QWORD *)(v4 + 32) = 0;
    *(_QWORD *)(v4 + 40) = 0;
    *(_QWORD *)(v4 + 48) = 0;
    *(_QWORD *)(v4 + 64) = 1;
    *(_QWORD *)(v4 + 72) = 0;
    *(_QWORD *)(v4 + 80) = 0;
    *(_QWORD *)(v4 + 96) = 0;
    *(_QWORD *)(v4 + 104) = 0;
    *(_QWORD *)(v4 + 120) = 1;
    *(_QWORD *)(v4 + 128) = 0;
    *(_QWORD *)(v4 + 136) = 0;
    *(_QWORD *)(v4 + 152) = 0;
    *(_QWORD *)(v4 + 160) = 0;
    *(_BYTE *)(v4 + 168) = 0;
    *(_QWORD *)(v4 + 176) = v4 + 192;
    *(_DWORD *)(v4 + 144) = 1065353216;
    if ( &v6[v8] && !v6 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v11[0] = v8;
    if ( v8 > 0xF )
    {
      v10 = sub_22409D0(v4 + 176, v11, 0);
      v5[22] = v10;
      v7 = (_BYTE *)v10;
      v5[24] = v11[0];
    }
    else
    {
      if ( v8 == 1 )
      {
        *(_BYTE *)(v4 + 192) = *v6;
LABEL_7:
        v5[23] = v8;
        v7[v8] = 0;
        v5[26] = a2;
        return v5;
      }
      if ( !v8 )
        goto LABEL_7;
    }
    memcpy(v7, v6, v8);
    v8 = v11[0];
    v7 = (_BYTE *)v5[22];
    goto LABEL_7;
  }
  return v5;
}
