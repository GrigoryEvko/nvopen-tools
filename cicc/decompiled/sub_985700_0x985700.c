// Function: sub_985700
// Address: 0x985700
//
__int64 __fastcall sub_985700(_BYTE *a1, _BYTE *a2)
{
  char v2; // al
  char v3; // al
  __int64 result; // rax
  _BYTE *v5; // r13
  __int64 v6; // rax
  char v7; // dl
  _BYTE *v8; // r13
  __int64 v9; // rax
  char v10; // dl
  _BYTE *v11; // r13
  __int64 v12; // rax
  char v13; // dl
  _BYTE *v14; // rax
  _BYTE *v15; // r12
  __int64 v16; // rax
  char v17; // dl
  _BYTE *v18; // rax

  v2 = *a1;
  if ( *a1 != 68 )
  {
LABEL_2:
    if ( v2 == 69 )
    {
      v5 = (_BYTE *)*((_QWORD *)a1 - 4);
      if ( *v5 == 82 )
      {
        v6 = sub_B53900(*((_QWORD *)a1 - 4));
        sub_B53630(v6, 32);
        if ( v7 )
        {
          if ( a2 == *((_BYTE **)v5 - 8) )
          {
            result = sub_9855A0(*((_QWORD *)v5 - 4));
            if ( (_BYTE)result )
              return result;
          }
        }
      }
    }
LABEL_3:
    v3 = *a2;
    if ( *a2 == 68 )
    {
      v11 = (_BYTE *)*((_QWORD *)a2 - 4);
      if ( *v11 != 82 )
        return 0;
      v12 = sub_B53900(*((_QWORD *)a2 - 4));
      sub_B53630(v12, 32);
      if ( v13 )
      {
        v14 = (_BYTE *)*((_QWORD *)v11 - 8);
        if ( a1 == v14 )
        {
          if ( v14 )
          {
            result = sub_9855A0(*((_QWORD *)v11 - 4));
            if ( (_BYTE)result )
              return result;
          }
        }
      }
      v3 = *a2;
    }
    if ( v3 == 69 )
    {
      v15 = (_BYTE *)*((_QWORD *)a2 - 4);
      if ( *v15 == 82 )
      {
        v16 = sub_B53900(v15);
        sub_B53630(v16, 32);
        if ( v17 )
        {
          v18 = (_BYTE *)*((_QWORD *)v15 - 8);
          if ( v18 )
          {
            if ( a1 == v18 )
            {
              result = sub_9855A0(*((_QWORD *)v15 - 4));
              if ( (_BYTE)result )
                return result;
            }
          }
        }
      }
    }
    return 0;
  }
  v8 = (_BYTE *)*((_QWORD *)a1 - 4);
  if ( *v8 != 82 )
    goto LABEL_3;
  v9 = sub_B53900(*((_QWORD *)a1 - 4));
  sub_B53630(v9, 32);
  if ( !v10 || a2 != *((_BYTE **)v8 - 8) || (result = sub_9855A0(*((_QWORD *)v8 - 4)), !(_BYTE)result) )
  {
    v2 = *a1;
    goto LABEL_2;
  }
  return result;
}
