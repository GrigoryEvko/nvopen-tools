// Function: sub_1003530
// Address: 0x1003530
//
unsigned __int8 *__fastcall sub_1003530(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r12
  char v7; // r14
  _BYTE *v8; // r13
  _QWORD *v9; // rax
  char v10; // dl
  _BYTE *v11; // rdx
  unsigned int v12; // r13d
  bool v13; // al
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  _BYTE *v19; // rax
  unsigned int v20; // ebx
  __int64 *v21; // rdi
  int v22; // r13d
  unsigned int v23; // r15d
  __int64 v24; // rax
  unsigned int v25; // r14d
  _QWORD *v26; // [rsp+8h] [rbp-58h]
  void *v27; // [rsp+8h] [rbp-58h]
  __int64 v28[10]; // [rsp+10h] [rbp-50h] BYREF

  v4 = a1;
  if ( *(_BYTE *)a1 == 13 || *(_BYTE *)a2 == 13 )
    return (unsigned __int8 *)v4;
  v7 = sub_1003090(a3, (unsigned __int8 *)a1);
  if ( !v7 )
  {
    if ( !a4 && (unsigned __int8)sub_1003090(a3, (unsigned __int8 *)a2) )
      return (unsigned __int8 *)v4;
    v8 = (_BYTE *)(a1 + 24);
    if ( *(_BYTE *)a1 == 18
      || (v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL) - 17, (unsigned int)v15 <= 1)
      && *(_BYTE *)a1 <= 0x15u
      && (v16 = sub_AD7630(a1, 0, v15)) != 0
      && (v8 = v16 + 24, *v16 == 18) )
    {
      v26 = *(_QWORD **)v8;
      v9 = sub_C33340();
      if ( v26 == v9 )
      {
        v10 = *(_BYTE *)(*((_QWORD *)v8 + 1) + 20LL) & 7;
        if ( v10 == 3 )
          return (unsigned __int8 *)v4;
      }
      else
      {
        v10 = v8[20] & 7;
        if ( v10 == 3 )
          return (unsigned __int8 *)v4;
      }
      if ( !v10 )
        return (unsigned __int8 *)v4;
      if ( a4 )
        return 0;
      v11 = v8;
      if ( v26 == v9 )
        v11 = (_BYTE *)*((_QWORD *)v8 + 1);
      if ( (v11[20] & 7) == 1 )
      {
        v27 = v9;
        if ( v9 == *(_QWORD **)v8 )
          sub_C3C790(v28, (_QWORD **)v8);
        else
          sub_C33EB0(v28, (__int64 *)v8);
        v21 = v28;
        if ( v27 == (void *)v28[0] )
          v21 = (__int64 *)v28[1];
        sub_C39170((__int64)v21);
        v4 = sub_AD8F10(*(_QWORD *)(v4 + 8), v28);
        sub_91D830(v28);
        return (unsigned __int8 *)v4;
      }
    }
    else if ( a4 )
    {
      return 0;
    }
    if ( *(_BYTE *)a2 == 17 )
    {
      v12 = *(_DWORD *)(a2 + 32);
      if ( v12 <= 0x40 )
        v13 = *(_QWORD *)(a2 + 24) == 0;
      else
        v13 = v12 == (unsigned int)sub_C444A0(a2 + 24);
    }
    else
    {
      v17 = *(_QWORD *)(a2 + 8);
      v18 = (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17;
      if ( (unsigned int)v18 > 1 || *(_BYTE *)a2 > 0x15u )
        return 0;
      v19 = sub_AD7630(a2, 0, v18);
      if ( !v19 || *v19 != 17 )
      {
        if ( *(_BYTE *)(v17 + 8) == 17 )
        {
          v22 = *(_DWORD *)(v17 + 32);
          if ( v22 )
          {
            v23 = 0;
            while ( 1 )
            {
              v24 = sub_AD69F0((unsigned __int8 *)a2, v23);
              if ( !v24 )
                break;
              if ( *(_BYTE *)v24 != 13 )
              {
                if ( *(_BYTE *)v24 != 17 )
                  return 0;
                v25 = *(_DWORD *)(v24 + 32);
                if ( v25 <= 0x40 )
                {
                  if ( *(_QWORD *)(v24 + 24) )
                    return 0;
                }
                else if ( v25 != (unsigned int)sub_C444A0(v24 + 24) )
                {
                  return 0;
                }
                v7 = 1;
              }
              if ( v22 == ++v23 )
              {
                if ( v7 )
                  return (unsigned __int8 *)v4;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v20 = *((_DWORD *)v19 + 8);
      if ( v20 <= 0x40 )
      {
        if ( !*((_QWORD *)v19 + 3) )
          return (unsigned __int8 *)v4;
        return 0;
      }
      v13 = v20 == (unsigned int)sub_C444A0((__int64)(v19 + 24));
    }
    if ( v13 )
      return (unsigned __int8 *)v4;
    return 0;
  }
  return sub_AD8F60(*(_QWORD *)(a1 + 8), 0, 0);
}
