// Function: sub_165BE20
// Address: 0x165be20
//
void __fastcall sub_165BE20(_BYTE *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // rdi
  unsigned __int8 v9; // al
  __int64 v10; // rdi
  _BYTE *v11; // rax

  v4 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    sub_16E2CE0(a2, v4);
    v6 = *(_BYTE **)(v4 + 24);
    if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 16) )
    {
      sub_16E7DE0(v4, 10);
    }
    else
    {
      *(_QWORD *)(v4 + 24) = v6 + 1;
      *v6 = 10;
    }
    v7 = *(_QWORD *)a1;
    a1[72] = 1;
    if ( v7 )
    {
      v8 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
      v9 = *(_BYTE *)(v8 + 16);
      if ( v9 > 0x17u && (v9 == 78 || v9 == 29) )
      {
        if ( v8 )
        {
          sub_155BD40(v8, v7, (__int64)(a1 + 16), 0);
          v10 = *(_QWORD *)a1;
          v11 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v11 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
            sub_16E7DE0(v10, 10);
          }
          else
          {
            *(_QWORD *)(v10 + 24) = v11 + 1;
            *v11 = 10;
          }
        }
      }
    }
  }
  else
  {
    a1[72] = 1;
  }
}
