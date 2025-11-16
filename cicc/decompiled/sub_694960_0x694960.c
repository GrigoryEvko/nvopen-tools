// Function: sub_694960
// Address: 0x694960
//
__int64 __fastcall sub_694960(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  __int64 v3; // rcx
  char v4; // si
  char v5; // dl
  char v6; // si
  int v7; // eax
  unsigned __int64 v8; // rsi
  int v9; // r8d
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rcx

  result = qword_4D03C50;
  *(_QWORD *)(qword_4D03C50 + 112LL) = a1;
  v2 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(result + 17) & 0x40) != 0 )
  {
LABEL_4:
    while ( *(_BYTE *)(v2 + 140) == 12 )
      v2 = *(_QWORD *)(v2 + 160);
    v3 = *(_QWORD *)(*(_QWORD *)v2 + 96LL);
    v4 = *(_BYTE *)(v3 + 181);
    v5 = v4 | 3;
    v6 = v4 | 1;
    *(_BYTE *)(v3 + 181) = v6;
    if ( (*(_BYTE *)(result + 17) & 0x40) == 0 )
      v5 = v6;
    *(_BYTE *)(v3 + 181) = v5;
    return result;
  }
  result = sub_693580();
  if ( (_DWORD)result )
  {
    v7 = sub_85E8D0();
    if ( v7 == -1 )
      goto LABEL_16;
    v8 = *(_QWORD *)(qword_4F04C68[0] + 776LL * v7 + 216);
    v9 = *((_DWORD *)qword_4F04C10 + 2);
    v10 = v9 & (v8 >> 3);
    v11 = (__int64 *)(*qword_4F04C10 + 16LL * v10);
    v12 = *v11;
    if ( *v11 != v8 )
    {
      while ( v12 )
      {
        v10 = v9 & (v10 + 1);
        v11 = (__int64 *)(*qword_4F04C10 + 16LL * v10);
        v12 = *v11;
        if ( v8 == *v11 )
          goto LABEL_14;
      }
LABEL_16:
      BUG();
    }
LABEL_14:
    result = *(_QWORD *)(**(_QWORD **)(v11[1] + 8) + 96LL);
    if ( (*(_BYTE *)(result + 181) & 1) != 0 )
    {
      result = qword_4D03C50;
      goto LABEL_4;
    }
  }
  return result;
}
