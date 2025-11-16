// Function: sub_86DAC0
// Address: 0x86dac0
//
__int64 __fastcall sub_86DAC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  char v5; // al
  __int64 *v6; // rax
  __int64 *v7; // r15
  __int64 v8; // rbx
  unsigned __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char *v14; // r14
  _QWORD *v15; // r15
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdi
  _QWORD *v23; // rax
  __int64 result; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rdx
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // [rsp+0h] [rbp-240h]
  __int64 v35; // [rsp+8h] [rbp-238h]
  int v36; // [rsp+10h] [rbp-230h]
  char v37; // [rsp+16h] [rbp-22Ah]
  bool v38; // [rsp+17h] [rbp-229h]
  __int64 *v39; // [rsp+28h] [rbp-218h] BYREF
  __int64 v40[66]; // [rsp+30h] [rbp-210h] BYREF

  v3 = 0xFFFFFFFFLL;
  v5 = *(_BYTE *)(a1 + 40);
  v39 = 0;
  v37 = v5;
  v38 = v5 == 16;
  v6 = sub_8600D0(0xFu, -1, 0, 0);
  v6[4] = a1;
  v7 = v6;
  v8 = sub_86B2C0(0);
  *(_QWORD *)(v8 + 24) = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(v8 + 56) = qword_4F06BC0;
  sub_86CBE0(v8);
  v9 = 9;
  v14 = (char *)sub_726700(9);
  **((_QWORD **)v14 + 7) = v7;
  *(_QWORD *)(a1 + 48) = v14;
  if ( a2 )
  {
    v15 = sub_726B30(0);
    v3 = 0;
    v9 = 0;
    *v15 = *(_QWORD *)sub_6E1A20(a2);
    v15[6] = sub_6B9820(0, 0, 0, 0, a2);
    *(_QWORD *)(*((_QWORD *)v14 + 7) + 24LL) = v15;
    if ( !dword_4F04C3C )
    {
      v3 = 21;
      v9 = (unsigned __int64)v15;
      sub_8699D0((__int64)v15, 21, 0);
    }
  }
  else if ( word_4F06418[0] != 75 || !unk_4D04858 )
  {
    v25 = 176LL * unk_4D03B90;
    *(_BYTE *)(qword_4D03B98 + v25 + 5) |= 0x80u;
    *(_QWORD *)(qword_4D03B98 + v25 + 136) = &v39;
    memset(v40, 0, 0x1D8u);
    v40[19] = (__int64)v40;
    v40[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v40[22]) |= 1u;
    BYTE4(v40[16]) |= 2u;
    sub_662DE0((unsigned int *)v40, 0);
    if ( word_4F06418[0] != 75 || !unk_4D04858 || (unsigned __int8)(*(_BYTE *)(a1 + 40) - 1) > 1u && !v38 )
    {
      sub_86DA30(v39, a1);
      v26 = 176LL * unk_4D03B90;
      *(_BYTE *)(qword_4D03B98 + v26 + 5) &= ~0x80u;
      *(_QWORD *)(qword_4D03B98 + v26 + 136) = 0;
      goto LABEL_9;
    }
    v35 = qword_4D03B98;
    v36 = unk_4D03B90;
    v30 = sub_726B30(20);
    *v30 = v40[3];
    *(_QWORD *)(*((_QWORD *)v14 + 7) + 24LL) = v30;
    if ( !dword_4F04C3C )
    {
      v34 = (__int64)v30;
      sub_86B430((__int64)v30);
      sub_869D70(v34, 21);
      v30 = (_QWORD *)v34;
    }
    v30[9] = **(_QWORD **)(v35 + 176LL * v36 + 136);
    v31 = 176LL * unk_4D03B90;
    v32 = v31 + qword_4D03B98;
    *(_BYTE *)(v32 + 5) &= ~0x80u;
    *(_QWORD *)(v32 + 144) = 0;
    *(_QWORD *)(qword_4D03B98 + v31 + 136) = 0;
    if ( !v40[34] )
      sub_6851C0(0xB1Fu, &v40[3]);
    v9 = (unsigned __int64)v39;
    v3 = a1;
    sub_86DA30(v39, a1);
    v33 = 176LL * unk_4D03B90;
    *(_BYTE *)(qword_4D03B98 + v33 + 5) &= ~0x80u;
    v11 = qword_4D03B98;
    *(_QWORD *)(qword_4D03B98 + v33 + 136) = 0;
  }
  if ( dword_4F077C4 != 2 || unk_4F07778 <= 201702 )
  {
    v11 = dword_4F077BC;
    if ( dword_4F077BC )
    {
      v10 = (unsigned int)dword_4F5FD5C;
      if ( !dword_4F5FD5C )
      {
        v9 = dword_4F063F8;
        if ( !sub_729F80(dword_4F063F8) )
        {
          v3 = (__int64)&dword_4F063F8;
          v9 = 2919;
          sub_684B30(0xB67u, &dword_4F063F8);
          dword_4F5FD5C = 1;
        }
      }
    }
  }
  sub_7B8B50(v9, (unsigned int *)v3, v10, v11, v12, v13);
  v16 = 176LL * unk_4D03B90;
  v17 = v16 + qword_4D03B98;
  v18 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 336);
  *(_BYTE *)(v17 + 5) |= 0x80u;
  v39 = 0;
  *(_QWORD *)(v17 + 144) = v18;
  *(_QWORD *)(qword_4D03B98 + v16 + 136) = &v39;
  if ( !(unsigned int)sub_679C10(0xAu) )
  {
    v27 = 176LL * unk_4D03B90;
    v28 = v27 + qword_4D03B98;
    *(_BYTE *)(v28 + 5) &= ~0x80u;
    v29 = qword_4D03B98;
    *(_QWORD *)(v28 + 144) = 0;
    *(_QWORD *)(v29 + v27 + 136) = 0;
    if ( v37 == 16 )
      v23 = (_QWORD *)sub_6B9750(1, 0);
    else
      v23 = (_QWORD *)sub_6D7680(0);
    goto LABEL_12;
  }
  memset(v40, 0, 0x1D8u);
  v40[19] = (__int64)v40;
  v40[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v40[22]) |= 1u;
  BYTE4(v40[16]) |= 2u;
  sub_662DE0((unsigned int *)v40, 0);
  sub_86DA30(v39, a1);
  v19 = 176LL * unk_4D03B90;
  v20 = v19 + qword_4D03B98;
  *(_BYTE *)(v20 + 5) &= ~0x80u;
  v21 = qword_4D03B98;
  *(_QWORD *)(v20 + 144) = 0;
  *(_QWORD *)(v21 + v19 + 136) = 0;
LABEL_9:
  v22 = sub_64FCE0(v40);
  if ( *(_BYTE *)(v22 + 177) == 2 )
    *(_QWORD *)(*((_QWORD *)v14 + 7) + 8LL) = *(_QWORD *)(v22 + 184);
  v23 = (_QWORD *)sub_696750((_QWORD *)v22, v38);
LABEL_12:
  *(_QWORD *)(*((_QWORD *)v14 + 7) + 16LL) = v23;
  *(_QWORD *)v14 = *v23;
  *(_QWORD *)(v14 + 36) = *(_QWORD *)(v8 + 24);
  result = *(_QWORD *)&dword_4F061D8;
  *(_QWORD *)(v14 + 44) = *(_QWORD *)&dword_4F061D8;
  return result;
}
