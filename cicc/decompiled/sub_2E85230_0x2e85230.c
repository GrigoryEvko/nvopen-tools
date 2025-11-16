// Function: sub_2E85230
// Address: 0x2e85230
//
_QWORD *__fastcall sub_2E85230(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r12
  _BYTE *v5; // rdi
  _BYTE *v6; // r14
  size_t v7; // r13
  __int64 v9; // rax
  size_t v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_22077B0(0xF0u);
  v4 = (_QWORD *)v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    v5 = (_BYTE *)(v3 + 224);
    *(_QWORD *)(v3 + 16) = &unk_50200EC;
    *(_QWORD *)(v3 + 56) = v3 + 104;
    *(_QWORD *)(v3 + 200) = a1;
    v6 = *(_BYTE **)a2;
    *(_QWORD *)(v3 + 112) = v3 + 160;
    v7 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)v3 = off_4A28FB8;
    *(_DWORD *)(v3 + 88) = 1065353216;
    *(_DWORD *)(v3 + 24) = 2;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 64) = 1;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 80) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_QWORD *)(v3 + 104) = 0;
    *(_QWORD *)(v3 + 120) = 1;
    *(_QWORD *)(v3 + 128) = 0;
    *(_QWORD *)(v3 + 136) = 0;
    *(_QWORD *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = 0;
    *(_BYTE *)(v3 + 168) = 0;
    *(_QWORD *)(v3 + 176) = 0;
    *(_QWORD *)(v3 + 184) = 0;
    *(_QWORD *)(v3 + 192) = 0;
    *(_QWORD *)(v3 + 208) = v3 + 224;
    *(_DWORD *)(v3 + 144) = 1065353216;
    if ( &v6[v7] && !v6 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v10[0] = v7;
    if ( v7 > 0xF )
    {
      v9 = sub_22409D0(v3 + 208, v10, 0);
      v4[26] = v9;
      v5 = (_BYTE *)v9;
      v4[28] = v10[0];
    }
    else
    {
      if ( v7 == 1 )
      {
        *(_BYTE *)(v3 + 224) = *v6;
LABEL_7:
        v4[27] = v7;
        v5[v7] = 0;
        return v4;
      }
      if ( !v7 )
        goto LABEL_7;
    }
    memcpy(v5, v6, v7);
    v7 = v10[0];
    v5 = (_BYTE *)v4[26];
    goto LABEL_7;
  }
  return v4;
}
