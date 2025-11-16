// Function: sub_7DD730
// Address: 0x7dd730
//
_DWORD *__fastcall sub_7DD730(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  _BYTE *v4; // rax
  char v5; // cl
  _BOOL4 v6; // r14d
  int v7; // r13d
  _QWORD *v8; // rbx
  __int64 v9; // rdi
  _DWORD *result; // rax
  int v11; // eax

  v3 = *(_QWORD *)(a1 + 168);
  v4 = *(_BYTE **)(v3 + 192);
  if ( !v4 )
  {
    v6 = 0;
    v7 = 1;
    if ( (*(_BYTE *)(a1 + 88) & 0x60) != 0 )
    {
      v11 = sub_8D23B0(a1);
      v3 = *(_QWORD *)(a1 + 168);
      v7 = v11 != 0;
      v6 = v11 == 0;
    }
    goto LABEL_4;
  }
  v5 = v4[136];
  LOBYTE(a2) = v5 == 2;
  if ( !v4[177] )
  {
    result = (_DWORD *)(v4[174] & 1);
    if ( !dword_4F077BC || v5 == 2 || (_BYTE)result )
    {
      if ( v5 == 2 )
      {
        v6 = 0;
        v7 = 1;
        goto LABEL_4;
      }
      v7 = dword_4D04848;
      if ( dword_4D04848 )
      {
        if ( !(_BYTE)result )
          return result;
        goto LABEL_13;
      }
    }
    else
    {
      if ( (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
      {
LABEL_13:
        v6 = 1;
        v7 = 0;
        goto LABEL_4;
      }
      result = &dword_4D04848;
      v7 = dword_4D04848;
      if ( dword_4D04848 )
        return result;
    }
    v6 = 1;
    goto LABEL_4;
  }
  v6 = v5 != 2;
  v7 = v5 == 2;
LABEL_4:
  v8 = *(_QWORD **)v3;
  if ( *(_QWORD *)v3 )
  {
    do
    {
      sub_7DC650(v8[5]);
      v9 = v8[5];
      if ( *(_BYTE *)(*(_QWORD *)(v9 + 152) + 136LL) == 1 )
        sub_7DD730(v9, a2);
      v8 = (_QWORD *)*v8;
    }
    while ( v8 );
  }
  return sub_7DCA00(a1, v7, v6);
}
