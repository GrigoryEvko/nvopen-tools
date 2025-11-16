// Function: sub_695660
// Address: 0x695660
//
__int64 __fastcall sub_695660(__int64 a1, int a2, __int64 *a3)
{
  __int64 v3; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rdi
  const char *v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v18; // rdi
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-3A8h] BYREF
  __int64 v26; // [rsp+10h] [rbp-3A0h] BYREF
  __int64 v27; // [rsp+18h] [rbp-398h] BYREF
  _BYTE v28[160]; // [rsp+20h] [rbp-390h] BYREF
  _BYTE v29[352]; // [rsp+C0h] [rbp-2F0h] BYREF
  _BYTE v30[400]; // [rsp+220h] [rbp-190h] BYREF

  v3 = a1;
  v25 = 0;
  v5 = *(_QWORD *)(unk_4F04C50 + 32LL);
  v6 = sub_71DF80(v5);
  if ( (*(_BYTE *)(v6 + 120) & 1) != 0 )
  {
    v10 = sub_7305B0();
    goto LABEL_14;
  }
  v7 = v6;
  if ( a2 )
  {
    v26 = *(_QWORD *)sub_6E1A20(a1);
    v8 = (__int64 *)sub_6E1A60(a1);
    v9 = *(_QWORD *)(v7 + 16);
    v27 = *v8;
    sub_6F8E70(v9, &v26, &v27, v29, 0);
    if ( v3 && !*(_BYTE *)(v3 + 8) )
    {
      unk_4D03C48 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 96LL);
      sub_832D70(v3, &v26);
    }
    sub_7029D0(v29, "yield_value", 0, v3, v29, v30);
    v10 = sub_6F6F40(v30, 0);
    goto LABEL_14;
  }
  sub_6E1DD0(&v25);
  sub_6E1E00(4, v28, 0, 0);
  v11 = *(_QWORD *)(v7 + 16);
  v26 = *a3;
  v27 = a3[1];
  sub_6F8E70(v11, &v26, &v27, v29, 0);
  v12 = "return_void";
  if ( v3 )
  {
    if ( !*(_BYTE *)(v3 + 8) )
    {
      unk_4D03C48 = *(_QWORD *)(*(_QWORD *)(v3 + 24) + 96LL);
      sub_832D70(v3, "return_void");
      if ( !*(_BYTE *)(v3 + 8) )
      {
        v23 = *(_QWORD *)(v3 + 24);
        if ( (unsigned int)sub_8D2600(*(_QWORD *)(v23 + 8)) )
        {
          v19 = sub_6F6F40(v23 + 8, 0);
          sub_6E1990(v3);
          sub_7029D0(v29, "return_void", 0, 0, v29, v30);
          v24 = sub_6F6F40(v30, 0);
          v14 = v24;
          if ( !*(_BYTE *)(v24 + 24) )
          {
            v3 = 0;
            goto LABEL_22;
          }
          v18 = *(_QWORD *)v24;
          v3 = 0;
          if ( (unsigned int)sub_8D2600(*(_QWORD *)v24) )
          {
LABEL_22:
            if ( v19 )
              v14 = sub_73DF90(v19, v14);
            goto LABEL_11;
          }
LABEL_18:
          v20 = *(_QWORD *)(v14 + 72);
          v21 = *(_BYTE *)(v20 + 24);
          if ( v21 == 20 )
          {
            sub_6854C0(0xC16u, (FILE *)(**(_QWORD **)(v20 + 56) + 48LL), **(_QWORD **)(v20 + 56));
          }
          else if ( v21 != 2 || (v22 = *(_QWORD *)(v20 + 56), *(_BYTE *)(v22 + 173) != 12) || *(_BYTE *)(v22 + 176) != 2 )
          {
            sub_721090(v18);
          }
          goto LABEL_22;
        }
      }
    }
    v12 = "return_value";
  }
  sub_7029D0(v29, v12, 0, v3, v29, v30);
  v13 = sub_6F6F40(v30, 0);
  v14 = v13;
  if ( *(_BYTE *)(v13 + 24) )
  {
    v18 = *(_QWORD *)v13;
    if ( !(unsigned int)sub_8D2600(*(_QWORD *)v13) )
    {
      v19 = 0;
      goto LABEL_18;
    }
  }
LABEL_11:
  v15 = 6;
  v10 = sub_6E2700(v14);
  v16 = sub_86E480(6, &dword_4F077C8);
  *(_QWORD *)(v16 + 72) = *(_QWORD *)(v7 + 48);
  if ( (*(_BYTE *)(v5 + 195) & 8) == 0 )
  {
    v15 = v5;
    *(_QWORD *)(v16 + 80) = *(_QWORD *)(sub_72B840(v5) + 88);
  }
  sub_6E2B30(v15, &dword_4F077C8);
  sub_6E1DF0(v25);
LABEL_14:
  sub_6E1990(v3);
  return v10;
}
