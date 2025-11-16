// Function: sub_7F47C0
// Address: 0x7f47c0
//
_DWORD *__fastcall sub_7F47C0(_QWORD *a1, int a2)
{
  __m128i *v3; // rsi
  _DWORD *result; // rax
  _QWORD *v5; // rbx
  __int64 i; // r14
  __int64 j; // r14
  __int64 k; // r14
  __int64 n; // rbx
  __int64 *v10; // rbx
  char m; // r14
  __int64 v12; // r15
  int v13; // [rsp+Ch] [rbp-A4h]
  __int64 v14; // [rsp+10h] [rbp-A0h]
  __int64 v15; // [rsp+18h] [rbp-98h]
  _BYTE v16[144]; // [rsp+20h] [rbp-90h] BYREF

  v15 = qword_4F06BC0;
  v14 = qword_4F04C50;
  if ( sub_7E16F0() )
  {
    v13 = dword_4F07270[0];
    dword_4D03F94 = 1;
    qword_4D03F68 = 0;
    qword_4F04C50 = 0;
    qword_4F06BC0 = *(_QWORD *)(qword_4F07288 + 88);
    unk_4D03F40 = 0;
    sub_7296B0(a2);
    unk_4F06CFC = 1;
    dword_4D03F90 = unk_4F073B8 == a2;
    sub_7E18E0((__int64)v16, qword_4F07288, 0);
    sub_7E64F0((__int64)a1);
    v3 = (__m128i *)dword_4D03F90;
    if ( dword_4D03F90 )
      sub_7DDED0((__int64)a1, (_QWORD *)dword_4D03F90);
    sub_7E9AF0((__int64)a1);
    if ( dword_4D03F90 )
    {
      v5 = (_QWORD *)qword_4F072C0;
      if ( qword_4F072C0 )
      {
        do
        {
          if ( (*(_DWORD *)(v5[1] + 192LL) & 0x8000400) == 0 )
          {
            for ( i = v5[3]; i; i = *(_QWORD *)(i + 112) )
            {
              while ( (unsigned int)sub_736DD0(i) )
              {
                i = *(_QWORD *)(i + 112);
                if ( !i )
                  goto LABEL_21;
              }
              sub_7EA690(i, v3);
            }
LABEL_21:
            for ( j = v5[4]; j; j = *(_QWORD *)(j + 112) )
            {
              if ( (*(_BYTE *)(j + 170) & 0x60) == 0 && *(_BYTE *)(j + 177) != 5 && (*(_BYTE *)(j - 8) & 8) == 0 )
                sub_7EC5C0(j, v3);
            }
            for ( k = v5[5]; k; k = *(_QWORD *)(k + 112) )
            {
              if ( (*(_BYTE *)(k + 124) & 1) == 0 )
                sub_7E9AF0(*(_QWORD *)(k + 128));
            }
          }
          v5 = (_QWORD *)*v5;
        }
        while ( v5 );
      }
      v10 = (__int64 *)&unk_4F06D00;
      for ( m = 0; m != 87; ++m )
      {
        v12 = *v10;
        if ( *v10 )
        {
          if ( m == 6 )
          {
            while ( 1 )
            {
              sub_7EA690(v12, v3);
LABEL_48:
              v12 = *(_QWORD *)(v12 - 16);
              if ( !v12 )
                break;
              if ( m != 6 )
                goto LABEL_45;
            }
          }
          else
          {
LABEL_45:
            if ( m == 7 )
            {
              if ( (*(_BYTE *)(v12 - 8) & 8) == 0 )
                sub_7EC5C0(v12, v3);
              goto LABEL_48;
            }
            if ( m == 2 )
            {
              sub_7EB190(v12, v3);
              goto LABEL_48;
            }
          }
        }
        v10 += 2;
      }
    }
    sub_7E4E40(a1);
    if ( dword_4D03F90 )
    {
      for ( n = *(_QWORD *)(qword_4F07288 + 168); n; n = *(_QWORD *)(n + 112) )
      {
        while ( (*(_BYTE *)(n + 124) & 1) != 0 )
        {
          n = *(_QWORD *)(n + 112);
          if ( !n )
            goto LABEL_37;
        }
        sub_7E4FA0(*(_QWORD **)(n + 128));
      }
LABEL_37:
      sub_806F60();
      if ( dword_4D04380 )
        sub_76FE50();
      sub_7DA790();
    }
    sub_7E0F10((__int64)a1);
    sub_7E1AA0();
    unk_4F06CFC = 0;
    qword_4F06BC0 = v15;
    qword_4F04C50 = v14;
    dword_4D03F94 = 0;
    if ( unk_4F073B8 == a2 )
      unk_4F072F8 = 1;
    return sub_7296B0(v13);
  }
  else
  {
    result = (_DWORD *)sub_7D7670();
    if ( (_DWORD)result )
      return sub_7DA8C0(a2);
  }
  return result;
}
