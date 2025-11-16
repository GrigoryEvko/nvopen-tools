// Function: sub_7D38C0
// Address: 0x7d38c0
//
__int64 __fastcall sub_7D38C0(__int64 a1, __int64 *a2)
{
  char v3; // al
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rax
  char v8; // dl
  __int64 *v9; // rax
  _QWORD *i; // r13
  __int64 v11; // rax
  char v12; // dl
  __int64 *v13[5]; // [rsp+8h] [rbp-28h] BYREF

  while ( 1 )
  {
    if ( a1 )
    {
      v3 = *(_BYTE *)(a1 + 140);
      if ( v3 == 8 )
      {
        while ( 1 )
        {
          v7 = sub_8D4050(a1);
          v8 = *(_BYTE *)(v7 + 140);
LABEL_16:
          if ( v8 == 12 )
          {
            do
              v7 = *(_QWORD *)(v7 + 160);
            while ( *(_BYTE *)(v7 + 140) == 12 );
          }
          if ( a1 == v7 )
            break;
          a1 = v7;
          v3 = *(_BYTE *)(v7 + 140);
          if ( v3 != 8 )
            goto LABEL_3;
        }
      }
      else
      {
LABEL_3:
        if ( v3 == 12 )
        {
          v7 = *(_QWORD *)(a1 + 160);
          v8 = *(_BYTE *)(v7 + 140);
          goto LABEL_16;
        }
        if ( v3 == 6 )
        {
          v7 = sub_8D46C0(a1);
          v8 = *(_BYTE *)(v7 + 140);
          goto LABEL_16;
        }
      }
    }
    result = *a2;
    if ( *a2 )
      break;
LABEL_22:
    v9 = (__int64 *)sub_8784C0();
    v9[1] = a1;
    *v9 = *a2;
    *a2 = (__int64)v9;
    result = *(unsigned __int8 *)(a1 + 140);
    if ( (unsigned __int8)result > 0xBu )
    {
      if ( (_BYTE)result != 13 )
        return result;
      v11 = sub_8D4870(a1);
      sub_7D38C0(v11, a2);
      a1 = sub_8D4890(a1);
    }
    else
    {
      if ( (unsigned __int8)result > 8u )
      {
        if ( (*(_BYTE *)(a1 + 177) & 0x10) == 0 )
          return result;
        result = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
        v13[0] = (__int64 *)result;
        if ( !result )
          return result;
        v12 = *(_BYTE *)(result + 8);
        if ( v12 != 3 )
          goto LABEL_33;
        sub_72F220(v13);
        result = (__int64)v13[0];
        if ( !v13[0] )
          return result;
        v12 = *((_BYTE *)v13[0] + 8);
LABEL_33:
        if ( !v12 )
          goto LABEL_38;
        while ( 1 )
        {
          do
          {
            result = *v13[0];
            v13[0] = (__int64 *)result;
            if ( !result )
              return result;
            v12 = *(_BYTE *)(result + 8);
            if ( v12 != 3 )
              goto LABEL_33;
            sub_72F220(v13);
            result = (__int64)v13[0];
            if ( !v13[0] )
              return result;
          }
          while ( *((_BYTE *)v13[0] + 8) );
LABEL_38:
          sub_7D38C0(*(_QWORD *)(result + 32), a2);
        }
      }
      if ( (_BYTE)result != 7 )
        return result;
      for ( i = **(_QWORD ***)(a1 + 168); i; i = (_QWORD *)*i )
        sub_7D38C0(i[1], a2);
      a1 = *(_QWORD *)(a1 + 160);
    }
  }
  while ( 1 )
  {
    v6 = *(_QWORD *)(result + 8);
    if ( v6 == a1 )
      return result;
    if ( v6 )
    {
      if ( a1 )
      {
        if ( dword_4F07588 )
        {
          v5 = *(_QWORD *)(v6 + 32);
          if ( *(_QWORD *)(a1 + 32) == v5 )
          {
            if ( v5 )
              return result;
          }
        }
      }
    }
    result = *(_QWORD *)result;
    if ( !result )
      goto LABEL_22;
  }
}
