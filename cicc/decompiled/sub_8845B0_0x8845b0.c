// Function: sub_8845B0
// Address: 0x8845b0
//
__int64 __fastcall sub_8845B0(int a1)
{
  __int64 result; // rax
  _QWORD *v2; // r12
  int v3; // r13d
  __int64 v4; // r14
  _QWORD *v5; // r15
  __int64 v6; // rbx
  char v7; // cl
  __int64 v8; // rcx
  __int64 v9; // rdi
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  char v15; // [rsp+9h] [rbp-47h]
  unsigned __int16 v16; // [rsp+Ah] [rbp-46h]
  unsigned int v17; // [rsp+Ch] [rbp-44h]
  __int64 v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  result = qword_4F04C68[0] + 776LL * a1;
  v2 = *(_QWORD **)(result + 456);
  v18 = result;
  if ( v2 )
  {
    v3 = 1;
    v4 = 0;
    v5 = 0;
    v19 = 0;
    v16 = word_4F077CC[0];
    v17 = dword_4F077C8;
    v15 = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 1;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = (__int64)v2;
        v2 = (_QWORD *)*v2;
        v7 = *(_BYTE *)(v6 + 53);
        *(_QWORD *)v6 = 0;
        *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) = v7 & 1
                                                                  | *(_BYTE *)(qword_4F04C68[0]
                                                                             + 776LL * dword_4F04C64
                                                                             + 13)
                                                                  & 0xFE;
        v8 = *(_QWORD *)(v6 + 24);
        if ( v8 )
          break;
        v13 = *(_QWORD *)(v6 + 16);
        v14 = *(_QWORD *)(v6 + 8);
        if ( v13 )
          v10 = sub_883A10(v14, v13, 0);
        else
          v10 = sub_884000(v14, 1);
LABEL_12:
        if ( v10 )
          goto LABEL_13;
        if ( (*(_BYTE *)(v18 + 7) & 8) == 0 )
        {
          sub_8774F0(
            *(_QWORD *)(v6 + 8),
            *(_QWORD *)(v6 + 24),
            (FILE *)(v6 + 32),
            *(_BYTE *)(v6 + 44),
            *(_DWORD *)(v6 + 48),
            *(unsigned __int8 *)(v6 + 52),
            0);
          v19 = *(_QWORD *)(v6 + 8);
          v17 = *(_DWORD *)(v6 + 32);
          v16 = *(_WORD *)(v6 + 36);
          goto LABEL_13;
        }
LABEL_4:
        if ( !v4 )
          v4 = v6;
        if ( v5 )
        {
          *v5 = v6;
          v3 = 0;
          v5 = (_QWORD *)v6;
        }
        else
        {
          v5 = (_QWORD *)v6;
          v3 = 0;
        }
        if ( !v2 )
        {
LABEL_15:
          *(_QWORD *)(v18 + 456) = v4;
          *(_QWORD *)(v18 + 464) = v5;
          result = qword_4F04C68[0] + 776LL * dword_4F04C64;
          *(_BYTE *)(result + 13) = v15 | *(_BYTE *)(result + 13) & 0xFE;
          return result;
        }
      }
      v9 = *(_QWORD *)(v6 + 8);
      if ( v9 != v19 )
        goto LABEL_11;
      v11 = *(unsigned int *)(v6 + 32);
      v12 = v17;
      if ( (_DWORD)v11 == v17 )
      {
        v12 = v16;
        v11 = *(unsigned __int16 *)(v6 + 36);
      }
      if ( v12 != v11 )
      {
LABEL_11:
        v10 = sub_8843A0(v9, *(_QWORD *)(v6 + 16), 0, v8, 0);
        goto LABEL_12;
      }
LABEL_13:
      if ( !v3 )
        goto LABEL_4;
      *(_QWORD *)v6 = qword_4F60008;
      qword_4F60008 = v6;
      if ( !v2 )
        goto LABEL_15;
    }
  }
  return result;
}
