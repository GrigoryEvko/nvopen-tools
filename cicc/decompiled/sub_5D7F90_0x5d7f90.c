// Function: sub_5D7F90
// Address: 0x5d7f90
//
void __fastcall sub_5D7F90(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h]
  __int64 v14; // [rsp+18h] [rbp-18h]

  if ( a1 )
    sub_5D6390(a1);
LABEL_3:
  if ( a2 )
  {
    while ( 1 )
    {
      if ( (unsigned int)sub_8D3410(a2[2]) )
      {
        putc(91, stream);
        v7 = a2[3];
        ++dword_4CF7F40;
        sub_5D32F0(v7);
        putc(93, stream);
        ++dword_4CF7F40;
        a2 = (_QWORD *)a2[1];
        goto LABEL_3;
      }
      v6 = a2[4];
      if ( (*(_BYTE *)(v6 + 145) & 0x10) != 0 )
        break;
      if ( !sub_5D7F20(v6) )
      {
        putc(46, stream);
        v8 = a2[4];
        ++dword_4CF7F40;
        sub_5D4E40(*(_BYTE **)(v8 + 8), v8);
      }
      qword_4CF7CB0 = 0;
      a2 = (_QWORD *)a2[1];
      qword_4CF7CA8 = 0;
      if ( !a2 )
        return;
    }
    v9 = qword_4CF7CA8;
    v13 = a2[4];
    v12 = qword_4CF7CA8;
    if ( qword_4CF7CB0 )
      *(_QWORD *)qword_4CF7CA8 = &v11;
    else
      qword_4CF7CB0 = (__int64)&v11;
    qword_4CF7CA8 = (__int64)&v11;
    v11 = 0;
    v14 = qword_4CF7CA0;
    v10 = a2[1];
    qword_4CF7CA0 += *(_QWORD *)(v6 + 128);
    sub_5D7F90(0, v10, v9, v3, v4, v5, 0, v12, v13, v14);
  }
}
