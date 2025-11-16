// Function: sub_6ABC20
// Address: 0x6abc20
//
__int64 __fastcall sub_6ABC20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // rax
  int v11; // edi
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r15
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rdx
  __int16 v26; // ax
  int v27; // [rsp+10h] [rbp-260h]
  __int64 v28; // [rsp+10h] [rbp-260h]
  __int64 v29; // [rsp+10h] [rbp-260h]
  __int16 v30; // [rsp+1Ah] [rbp-256h]
  int v31; // [rsp+1Ch] [rbp-254h]
  char v32; // [rsp+20h] [rbp-250h] BYREF
  unsigned int v33; // [rsp+24h] [rbp-24Ch] BYREF
  __int64 v34; // [rsp+28h] [rbp-248h] BYREF
  __int64 v35; // [rsp+30h] [rbp-240h] BYREF
  __int64 v36; // [rsp+38h] [rbp-238h] BYREF
  _BYTE v37[160]; // [rsp+40h] [rbp-230h] BYREF
  _BYTE v38[400]; // [rsp+E0h] [rbp-190h] BYREF

  v34 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( a1 )
  {
    sub_6F8810(
      a1,
      (unsigned int)&v33,
      (unsigned int)v38,
      (unsigned int)&v35,
      (unsigned int)&v36,
      (unsigned int)&v32,
      (__int64)v37);
    v31 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
    v30 = *(_WORD *)(*(_QWORD *)a1 + 48LL);
    v9 = sub_687860(a1, &v33);
    sub_68B050(v9, (__int64)&v33, &v35);
    sub_6E2140(5, v37, 0, 0, a1);
    v10 = qword_4D03C50;
    v11 = qword_4F077B4;
    *(_WORD *)(qword_4D03C50 + 18LL) |= 0x8020u;
    if ( v11 )
      *(_BYTE *)(v10 + 20) |= 1u;
  }
  else
  {
    v36 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, v7, v8);
    v22 = sub_687860(0, &v33);
    sub_68B050(v22, (__int64)&v33, &v35);
    sub_6E2140(5, v37, 0, 0, 0);
    v23 = qword_4D03C50;
    v24 = qword_4F077B4;
    *(_WORD *)(qword_4D03C50 + 18LL) |= 0x8020u;
    if ( v24 )
      *(_BYTE *)(v23 + 20) |= 1u;
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    ++*(_QWORD *)(qword_4D03C50 + 40LL);
    sub_69ED20((__int64)v38, 0, 0, 0);
  }
  sub_6F6C80(v38);
  v12 = sub_6F6F40(v38, 0);
  v15 = sub_6E2700(v12);
  if ( (dword_4F04C44 != -1
     || (v16 = qword_4F04C68, v17 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v17 + 6) & 6) != 0)
     || *(_BYTE *)(v17 + 4) == 12)
    && (unsigned int)sub_731EE0(v15) )
  {
    sub_724C70(v34, 12);
    sub_7249B0(v34, 10);
    v25 = v34;
    *(_QWORD *)(v34 + 192) = v15;
    *(_QWORD *)(v25 + 128) = sub_72C390();
  }
  else
  {
    v27 = sub_731B40(v15, 0, v16, v13, v14);
    v18 = sub_72C390();
    sub_72BAF0(v34, v27 == 0, *(unsigned __int8 *)(v18 + 160));
    v28 = v34;
    *(_QWORD *)(v28 + 128) = sub_72C390();
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
    {
      v21 = sub_72C390();
      v29 = v34;
      *(_QWORD *)(v29 + 144) = sub_73DBF0(24, v21, v15);
    }
  }
  sub_6E6A50(v34, a2);
  if ( !a1 )
  {
    v26 = WORD2(qword_4F063F0);
    v31 = qword_4F063F0;
    unk_4F061D8 = qword_4F063F0;
    v30 = v26;
    unk_4F061DC = v26;
    sub_7BE280(28, 18, 0, 0);
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    --*(_QWORD *)(qword_4D03C50 + 40LL);
  }
  *(_DWORD *)(a2 + 68) = v36;
  *(_WORD *)(a2 + 72) = WORD2(v36);
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a2 + 68);
  *(_DWORD *)(a2 + 76) = v31;
  *(_WORD *)(a2 + 80) = v30;
  unk_4F061D8 = *(_QWORD *)(a2 + 76);
  sub_6E3280(a2, &v36);
  sub_6E3BA0(a2, &v36, 0, 0);
  sub_6E2B30(a2, &v36);
  v19 = v35;
  sub_729730(v33);
  qword_4F06BC0 = v19;
  return sub_724E30(&v34);
}
