// Function: sub_6382A0
// Address: 0x6382a0
//
__int64 __fastcall sub_6382A0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx

  *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 14) |= 8u;
  if ( word_4F06418[0] == 27 )
    goto LABEL_11;
  if ( dword_4D04428 )
  {
    if ( word_4F06418[0] != 73 )
    {
      v7 = 2333;
      *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
      v8 = qword_4F061C8;
      ++*(_BYTE *)(qword_4F061C8 + 35LL);
      ++*(_BYTE *)(v8 + 81);
      goto LABEL_5;
    }
LABEL_11:
    sub_7B80F0();
    if ( !a2 )
    {
      sub_7BDC10(a1, 0, a3);
      sub_7B8B50(a1, 0, v12, v13);
      sub_7B8160();
      goto LABEL_10;
    }
    if ( word_4F06418[0] == 27 )
    {
      sub_637AB0(a1, (__int64)a2, a3, a4);
    }
    else
    {
      if ( !a4 )
        a4 = a3;
      sub_637960(a1, a4, (__int64)a2, 0);
    }
    sub_7B8160();
    goto LABEL_9;
  }
  v7 = 125;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  ++*(_BYTE *)(qword_4F061C8 + 35LL);
LABEL_5:
  sub_6851D0(v7);
  v10 = qword_4F061C8;
  if ( dword_4D04428 )
    --*(_BYTE *)(qword_4F061C8 + 81LL);
  --*(_BYTE *)(v10 + 35);
  if ( a2 )
  {
    a2[3] = sub_72C9D0(v7, a2, v9);
LABEL_9:
    a2[5] = *a5;
    a2[6] = unk_4F061D8;
  }
LABEL_10:
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_BYTE *)(result + 14) &= ~8u;
  return result;
}
