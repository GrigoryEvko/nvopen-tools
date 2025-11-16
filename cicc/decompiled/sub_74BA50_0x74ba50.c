// Function: sub_74BA50
// Address: 0x74ba50
//
__int64 __fastcall sub_74BA50(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 (*v3)(void); // rax
  __int64 result; // rax
  __int64 **v5; // rbx
  bool v6; // r14
  void (__fastcall *v7)(const char *, __int64); // rax
  unsigned __int8 v8; // al
  __int64 *v9; // r13
  unsigned __int8 *v10; // rbx
  void (__fastcall *v11)(const char *, __int64); // rdx
  __int64 i; // rax
  int v13; // r13d
  _QWORD *v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // [rsp+Ch] [rbp-34h]

  v2 = a1;
  v3 = *(__int64 (**)(void))(a2 + 64);
  if ( v3 )
    return v3();
  v5 = *(__int64 ***)(a1 + 168);
  v17 = 0;
  v6 = (*((_BYTE *)v5 + 17) & 4) != 0;
  if ( v5[5] )
    v17 = (*((_BYTE *)v5 + 18) | (unsigned __int8)(*((_WORD *)v5 + 9) >> 7)) & 0x7F;
  (*(void (__fastcall **)(char *, __int64))a2)("(", a2);
  if ( ((_BYTE)v5[2] & 6) != 2 && (dword_4F072C8 || *(_BYTE *)(a2 + 136)) )
  {
LABEL_8:
    v7 = *(void (__fastcall **)(const char *, __int64))a2;
    goto LABEL_9;
  }
  v9 = *v5;
  if ( *v5 )
  {
    while ( 1 )
    {
      if ( !*(_BYTE *)(a2 + 155) )
        goto LABEL_30;
      while ( *((_DWORD *)v9 + 9) )
      {
LABEL_30:
        sub_74B930(v9[1], a2);
        if ( (*((_BYTE *)v9 + 33) & 1) != 0 )
          (*(void (__fastcall **)(char *, __int64))a2)("...", a2);
        if ( !*v9 )
        {
LABEL_49:
          if ( ((_BYTE)v5[2] & 1) != 0 )
            (*(void (__fastcall **)(const char *, __int64))a2)(", ...", a2);
          goto LABEL_8;
        }
        if ( !*(_BYTE *)(a2 + 155) || *(_DWORD *)(*v9 + 36) )
        {
          (*(void (__fastcall **)(char *, __int64))a2)(", ", a2);
          break;
        }
        v9 = (__int64 *)*v9;
      }
      v9 = (__int64 *)*v9;
      if ( !v9 )
        goto LABEL_49;
    }
  }
  v7 = *(void (__fastcall **)(const char *, __int64))a2;
  if ( ((_BYTE)v5[2] & 1) != 0 )
  {
    ((void (__fastcall *)(char *, __int64, _QWORD))v7)("...", a2, (_BYTE)v5[2] & 1);
    (*(void (__fastcall **)(char *, __int64))a2)(")", a2);
    goto LABEL_10;
  }
  if ( dword_4F072C8 == 1 )
  {
    v7("void", a2);
    (*(void (__fastcall **)(char *, __int64))a2)(")", a2);
    goto LABEL_10;
  }
LABEL_9:
  v7(")", a2);
LABEL_10:
  if ( !*(_BYTE *)(a2 + 136) )
  {
    v8 = (*((_BYTE *)v5 + 17) >> 4) & 7;
    if ( v8 > 1u )
    {
      v13 = v8;
      if ( !(unsigned int)sub_8D7260(v8, unk_4F06CF8, 0) )
      {
        (*(void (__fastcall **)(char *, __int64))a2)(" ", a2);
        (*(void (__fastcall **)(_QWORD, __int64))a2)(*(&off_4B6EAE0 + v13), a2);
      }
    }
  }
  if ( v6 )
  {
    if ( v5[5] )
    {
      if ( (v17 & 1) == 0 )
        (*(void (__fastcall **)(const char *, __int64))a2)(" mutable", a2);
    }
    else
    {
      (*(void (__fastcall **)(const char *, __int64))a2)(" static", a2);
    }
  }
  else if ( v17 )
  {
    (*(void (__fastcall **)(char *, __int64))a2)(" ", a2);
    sub_746940(v17, -1, 0, a2);
  }
  result = *((_BYTE *)v5 + 19) & 0xC0;
  if ( (*((_BYTE *)v5 + 19) & 0xC0) == 0x40 )
  {
    result = (*(__int64 (__fastcall **)(char *, __int64))a2)(" &", a2);
  }
  else if ( (*((_BYTE *)v5 + 19) & 0xC0) == 0x80 )
  {
    result = (*(__int64 (__fastcall **)(const char *, __int64))a2)(" &&", a2);
  }
  if ( ((_BYTE)v5[2] & 8) != 0 || v6 )
  {
    if ( *(_BYTE *)(a2 + 136) && *(_BYTE *)(a2 + 141) )
      return result;
    (*(void (__fastcall **)(char *, __int64))a2)("->", a2);
    result = sub_74B930(*(_QWORD *)(a1 + 160), a2);
  }
  if ( !*(_BYTE *)(a2 + 136) )
  {
    result = (unsigned int)dword_4F06978;
    if ( dword_4F06978 )
    {
      while ( *(_BYTE *)(v2 + 140) == 12 )
        v2 = *(_QWORD *)(v2 + 160);
      result = *(_QWORD *)(v2 + 168);
      v10 = *(unsigned __int8 **)(result + 56);
      if ( v10 )
      {
        result = *v10;
        if ( (result & 6) == 0 )
        {
          v11 = *(void (__fastcall **)(const char *, __int64))a2;
          if ( (result & 1) == 0 )
          {
            v14 = (_QWORD *)*((_QWORD *)v10 + 1);
            v11(" throw(", a2);
            for ( ; v14; v14 = (_QWORD *)*v14 )
            {
              sub_74B930(v14[1], a2);
              if ( !*v14 )
                break;
              (*(void (__fastcall **)(char *, __int64))a2)(", ", a2);
            }
            return (*(__int64 (__fastcall **)(char *, __int64))a2)(")", a2);
          }
          v11(" noexcept", a2);
          for ( result = *v10; (result & 0x40) != 0; result = *v10 )
          {
            for ( i = *(_QWORD *)(*((_QWORD *)v10 + 1) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            v10 = *(unsigned __int8 **)(*(_QWORD *)(i + 168) + 56LL);
          }
          if ( (result & 0x20) != 0 )
            return (*(__int64 (__fastcall **)(const char *, __int64))a2)("(<expr>)", a2);
          if ( *((_QWORD *)v10 + 1) )
          {
            (*(void (__fastcall **)(char *, __int64))a2)("(", a2);
            sub_748000(*((_QWORD *)v10 + 1), 0, a2, v15, v16);
            return (*(__int64 (__fastcall **)(char *, __int64))a2)(")", a2);
          }
        }
      }
    }
  }
  return result;
}
