// Function: sub_76C540
// Address: 0x76c540
//
__int64 __fastcall sub_76C540(__int64 a1, __int64 (__fastcall *a2)(_QWORD))
{
  __int64 i; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 result; // rax
  _QWORD *k; // r12
  _QWORD *v9; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 j; // r13
  __int64 v15; // rdi
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 28) )
    goto LABEL_2;
  result = (__int64)&qword_4F07280;
  v9 = (_QWORD *)qword_4F072C0;
  if ( qword_4F072C0 )
  {
    v10 = qword_4F04C50;
    do
    {
      v13 = *(int *)(v9[1] + 160LL);
      if ( (_DWORD)v13 )
      {
        v11 = unk_4F072B8 + 16 * v13;
        v12 = *(_QWORD *)(unk_4F073B0 + 8LL * *(int *)(v11 + 8));
        if ( v12 )
          v12 = *(_QWORD *)v11;
        qword_4F04C50 = v12;
      }
      else
      {
        qword_4F04C50 = 0;
      }
      result = a2(v9[3]);
      qword_4F04C50 = v10;
      v9 = (_QWORD *)*v9;
    }
    while ( v9 );
  }
  if ( dword_4F077C4 == 2 )
  {
LABEL_2:
    for ( i = *(_QWORD *)(a1 + 144); i; qword_4F04C50 = v18 )
    {
      while ( 1 )
      {
        v4 = *(int *)(i + 160);
        if ( (_DWORD)v4 )
        {
          if ( *(_QWORD *)(unk_4F073B0 + 8LL * *(int *)(i + 164)) )
          {
            v5 = unk_4F072B8 + 16 * v4;
            if ( *(_QWORD *)(unk_4F073B0 + 8LL * *(int *)(v5 + 8)) )
            {
              v6 = *(_QWORD *)v5;
              if ( *(_QWORD *)v5 )
              {
                if ( (*(_BYTE *)(v6 + 29) & 1) == 0 )
                  break;
              }
            }
          }
        }
        i = *(_QWORD *)(i + 112);
        if ( !i )
          goto LABEL_9;
      }
      v17 = qword_4F04C50;
      qword_4F04C50 = v6;
      v18 = v17;
      sub_76C540(v6, a2);
      i = *(_QWORD *)(i + 112);
    }
LABEL_9:
    result = *(unsigned __int8 *)(a1 + 28);
    if ( (((_BYTE)result - 15) & 0xFD) == 0 || (_BYTE)result == 2 )
      result = a2(*(_QWORD *)(a1 + 104));
    if ( dword_4F077C4 == 2 )
    {
      for ( j = *(_QWORD *)(a1 + 104); j; j = *(_QWORD *)(j + 112) )
      {
        while ( 1 )
        {
          result = (unsigned int)*(unsigned __int8 *)(j + 140) - 9;
          if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) <= 2u )
          {
            result = *(_QWORD *)(j + 168);
            if ( result )
            {
              v15 = *(_QWORD *)(result + 152);
              if ( v15 )
              {
                if ( (*(_BYTE *)(v15 + 29) & 0x20) == 0 )
                  break;
              }
            }
          }
          j = *(_QWORD *)(j + 112);
          if ( !j )
            goto LABEL_32;
        }
        result = sub_76C540(v15, a2);
      }
LABEL_32:
      v16 = *(_QWORD *)(a1 + 168);
      if ( v16 )
      {
        if ( (*(_BYTE *)(v16 + 124) & 1) == 0 )
          goto LABEL_36;
        while ( 1 )
        {
          v16 = *(_QWORD *)(v16 + 112);
          if ( !v16 )
            break;
          if ( (*(_BYTE *)(v16 + 124) & 1) == 0 )
LABEL_36:
            result = sub_76C540(*(_QWORD *)(v16 + 128), a2);
        }
      }
    }
    for ( k = *(_QWORD **)(a1 + 160); k; k = (_QWORD *)*k )
      result = sub_76C540(k, a2);
  }
  return result;
}
