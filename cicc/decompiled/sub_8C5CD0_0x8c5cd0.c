// Function: sub_8C5CD0
// Address: 0x8c5cd0
//
_DWORD *sub_8C5CD0()
{
  _QWORD *v0; // rdi
  _QWORD *v1; // rbx
  _QWORD *v2; // rbx
  __int64 v3; // r13
  __int64 *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 i; // rcx
  __int64 j; // rdx
  __int64 v9; // rax
  _QWORD *v10; // r15
  __int64 v11; // rbx
  _QWORD *v12; // rdi
  _QWORD *k; // rax
  _QWORD *v14; // rdx
  _DWORD *result; // rax
  int v16; // r15d
  __int64 *(__fastcall *v17)(__int64 *, unsigned __int8); // r14
  __int64 v18; // rbx
  int v19; // edi
  __int64 v20; // rdx
  _DWORD v21[13]; // [rsp+Ch] [rbp-34h] BYREF

  dword_4F6023C = 1;
  dword_4F60238 = 0;
  v0 = qword_4D03FD0;
  v1 = (_QWORD *)*qword_4D03FD0;
  if ( *qword_4D03FD0 )
  {
    do
    {
      v21[0] = 0;
      sub_8D0910(v1);
      sub_8C4EC0(qword_4F07288, v21);
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
    dword_4F6023C = 0;
    v0 = qword_4D03FD0;
    v2 = (_QWORD *)*qword_4D03FD0;
    if ( *qword_4D03FD0 )
    {
      do
      {
        sub_8D0910(v2);
        v3 = qword_4F07288;
        sub_759B50(
          (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C2C50,
          (__int64 (__fastcall *)(_QWORD, _QWORD, _QWORD))sub_8C3290,
          0,
          0,
          (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C37B0,
          0);
        sub_8C3A50(v3);
        sub_8C3020(v3);
        v2 = (_QWORD *)*v2;
      }
      while ( v2 );
      v0 = qword_4D03FD0;
    }
  }
  else
  {
    dword_4F6023C = 0;
  }
  sub_8D0910(v0);
  v4 = (__int64 *)*qword_4D03FD0;
  if ( *qword_4D03FD0 )
  {
    do
    {
      sub_8C4010(v4[1]);
      v5 = v4[25];
      if ( v5 )
      {
        if ( (*(_BYTE *)(v5 - 8) & 3) == 3 )
        {
          sub_8C3650((__int64 *)v4[25], 0xBu, 0);
          v5 = *(_QWORD *)(v5 - 24);
          if ( (*(_BYTE *)(v5 - 8) & 2) != 0 )
            v5 = *(_QWORD *)(v5 - 24);
        }
        unk_4F07290 = v5;
      }
      if ( *((_BYTE *)v4 + 299) )
        unk_4F072F3 = 1;
      if ( *((_BYTE *)v4 + 300) )
        unk_4F072F4 = 1;
      if ( *((_BYTE *)v4 + 301) )
        unk_4F072F5 = 1;
      v6 = qword_4F07300;
      for ( i = 0; v6; v6 = *(_QWORD *)(v6 + 112) )
        i = v6;
      for ( j = v4[39]; j; j = *(_QWORD *)(j + 112) )
      {
        while ( 1 )
        {
          v9 = *(_QWORD *)(j - 24);
          if ( (*(_BYTE *)(j - 8) & 8) == 0 )
            break;
          v9 = *(_QWORD *)(v9 - 24);
          if ( !*(_QWORD *)(v9 + 112) && v9 != i )
            break;
          j = *(_QWORD *)(j + 112);
          if ( !j )
            goto LABEL_27;
        }
        if ( i )
          *(_QWORD *)(i + 112) = v9;
        else
          qword_4F07300 = v9;
        *(_BYTE *)(v9 + 141) |= 0x40u;
        i = v9;
        *(_QWORD *)(v9 + 112) = 0;
      }
LABEL_27:
      sub_8C32F0((_QWORD *)v4[1]);
      v10 = (_QWORD *)v4[31];
      if ( v10 )
      {
        do
        {
          while ( 1 )
          {
            if ( dword_4F077C4 == 2 )
            {
              v11 = v10[3];
              if ( v11 )
                break;
            }
            v10 = (_QWORD *)*v10;
            if ( !v10 )
              goto LABEL_37;
          }
          do
          {
            if ( (unsigned __int8)(*(_BYTE *)(v11 + 140) - 9) <= 2u )
            {
              v12 = *(_QWORD **)(*(_QWORD *)(v11 + 168) + 152LL);
              if ( v12 )
                sub_8C32F0(v12);
            }
            v11 = *(_QWORD *)(v11 + 112);
          }
          while ( v11 );
          v10 = (_QWORD *)*v10;
        }
        while ( v10 );
LABEL_37:
        for ( k = (_QWORD *)v4[31]; k; k = (_QWORD *)*k )
        {
          if ( !*(_DWORD *)(k[1] + 160LL) )
          {
            v14 = (_QWORD *)*(k - 3);
            if ( qword_4F072C0 )
              **(_QWORD **)(qword_4D03FF0 + 352) = v14;
            else
              qword_4F072C0 = *(k - 3);
            *v14 = 0;
            *(_QWORD *)(qword_4D03FF0 + 352) = v14;
          }
        }
      }
      v4 = (__int64 *)*v4;
    }
    while ( v4 );
    if ( dword_4F077C4 != 2 )
      goto LABEL_48;
  }
  else
  {
    result = &dword_4F077C4;
    if ( dword_4F077C4 != 2 )
      return result;
  }
  dword_4F60238 = 1;
  v16 = 2;
  v17 = sub_8C39E0;
  while ( 1 )
  {
    sub_759B50(
      0,
      0,
      (__int64 (__fastcall *)(_QWORD, _QWORD))v17,
      (__int64 (__fastcall *)(_QWORD, _QWORD))v17,
      (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C3220,
      0);
    if ( dword_4F073A8 > 1 )
    {
      v18 = 0;
      v19 = 2;
      do
      {
        if ( *((_QWORD *)qword_4F073B0 + v18 + 2) )
        {
          v20 = *((_QWORD *)qword_4F072B0 + v18 + 2);
          if ( (*(_BYTE *)(v20 - 8) & 2) == 0 )
          {
            if ( *(_BYTE *)(v20 + 28) )
              sub_75AFC0(
                v19,
                0,
                0,
                (__int64 (__fastcall *)(_QWORD, _QWORD))v17,
                (__int64 (__fastcall *)(_QWORD, _QWORD))v17,
                (__int64 (__fastcall *)(_QWORD, _QWORD))sub_8C3220,
                0);
          }
        }
        v19 = ++v18 + 2;
      }
      while ( dword_4F073A8 >= (int)v18 + 2 );
    }
    v17 = 0;
    if ( v16 == 1 )
      break;
    v16 = 1;
  }
  dword_4F60238 = 0;
  sub_8C3410(qword_4F07288);
LABEL_48:
  result = qword_4D03FD0;
  if ( *qword_4D03FD0 )
    return (_DWORD *)sub_72B900();
  return result;
}
