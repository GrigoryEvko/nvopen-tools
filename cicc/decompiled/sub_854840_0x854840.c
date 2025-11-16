// Function: sub_854840
// Address: 0x854840
//
_QWORD *__fastcall sub_854840(unsigned __int8 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rcx
  __int64 v5; // r9
  __int64 v6; // r15
  _QWORD **v7; // r14
  _QWORD *v8; // r12
  _QWORD *v9; // r13
  _QWORD *v10; // rbx
  _QWORD *v11; // r8
  _QWORD *v12; // rax
  __int64 v16; // [rsp+20h] [rbp-40h]
  _QWORD *v17; // [rsp+28h] [rbp-38h]

  v4 = qword_4D03D40[a1];
  v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v6 = v5;
  v7 = (_QWORD **)(v5 + 432);
  if ( *(_DWORD *)(v4 + 12) == 1 )
  {
    a4 |= 1u;
    v7 = (_QWORD **)(v5 + 440);
  }
  v8 = 0;
  v9 = 0;
  while ( 1 )
  {
    v10 = *v7;
    if ( *v7 )
    {
      v11 = 0;
      do
      {
        while ( 1 )
        {
          v12 = v10;
          v10 = (_QWORD *)*v10;
          if ( v12[1] == v4 )
            break;
          v11 = v12;
LABEL_7:
          if ( !v10 )
            goto LABEL_15;
        }
        if ( v11 )
          *v11 = v10;
        else
          *v7 = v10;
        if ( v9 )
          *v8 = v12;
        else
          v9 = v12;
        *v12 = 0;
        v8 = v12;
        if ( (*(_BYTE *)(v4 + 17) & 8) == 0 )
          goto LABEL_7;
        v16 = v4;
        v17 = v11;
        sub_8543B0(v12, a2, a3);
        v11 = v17;
        v4 = v16;
      }
      while ( v10 );
    }
LABEL_15:
    if ( a4 || qword_4F04C68[0] == v6 )
      return v9;
    if ( *(_BYTE *)(v6 + 4) == 9 )
      v6 = qword_4F04C68[0];
    else
      v6 -= 776;
    v7 = (_QWORD **)(v6 + 432);
  }
}
