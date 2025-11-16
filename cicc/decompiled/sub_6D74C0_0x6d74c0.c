// Function: sub_6D74C0
// Address: 0x6d74c0
//
__int64 __fastcall sub_6D74C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned int v7; // r14d
  char v8; // dl
  int v9; // r12d
  __int16 v10; // bx
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v18; // [rsp-8h] [rbp-258h]
  unsigned __int16 v19; // [rsp+Eh] [rbp-242h]
  __int64 v20; // [rsp+10h] [rbp-240h] BYREF
  __int64 v21; // [rsp+18h] [rbp-238h] BYREF
  char v22[160]; // [rsp+20h] [rbp-230h] BYREF
  _QWORD v23[9]; // [rsp+C0h] [rbp-190h] BYREF
  int v24; // [rsp+10Ch] [rbp-144h]
  __int16 v25; // [rsp+110h] [rbp-140h]

  v6 = a1;
  v20 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  sub_6E1DD0(&v21);
  sub_6E1E00(word_4D04898 == 0 ? 1 : 3, v22, 0, 0);
  sub_6E2170(v21);
  v7 = dword_4F063F8;
  v19 = word_4F063FC[0];
  if ( HIDWORD(qword_4F077B4) && !word_4D04898 )
    sub_6BA150(0, 0, 0, 0, 0, (__int64)v23, 0, 0);
  else
    sub_69ED20((__int64)v23, 0, 0, 1);
  v8 = *(_BYTE *)(a1 + 140);
  v9 = v24;
  v10 = v25;
  if ( v8 == 12 )
  {
    v11 = a1;
    do
    {
      v11 = *(_QWORD *)(v11 + 160);
      v8 = *(_BYTE *)(v11 + 140);
    }
    while ( v8 == 12 );
  }
  if ( !v8 )
    v6 = 0;
  v12 = v6;
  sub_697340(v23, v6, 0xC1u, 0, 0, 0, v20);
  sub_6E2AC0(v20);
  if ( *(_BYTE *)(v20 + 173) )
  {
    v15 = sub_724E50(&v20, v6, v18, v13, v14);
    *(_DWORD *)(v15 + 64) = v7;
    v16 = v15;
    *(_DWORD *)(v15 + 112) = v9;
    *(_WORD *)(v15 + 68) = v19;
    *(_WORD *)(v15 + 116) = v10;
  }
  else
  {
    sub_724E30(&v20);
    v16 = 0;
  }
  dword_4F061D8 = v9;
  unk_4F061DC = v10;
  sub_6E2B30(&v20, v12);
  sub_6E1DF0(v21);
  return v16;
}
