// Function: sub_31F1C50
// Address: 0x31f1c50
//
__int64 __fastcall sub_31F1C50(_BYTE **a1, __int64 a2, __int64 a3, char *a4, __int64 a5)
{
  unsigned int v5; // r13d
  char v6; // al
  __int64 v7; // r14
  char v8; // al
  __int64 (*v10)(); // rax
  void (__fastcall *v11)(__int64, __int64, __int64); // rax
  __int64 v12; // rax

  v5 = 1;
  if ( !a4 )
    return v5;
  v6 = *a4;
  if ( !*a4 || a4[1] )
    return v5;
  v7 = *(_QWORD *)(a2 + 32) + 40LL * (unsigned int)a3;
  if ( v6 == 110 )
  {
    if ( *(_BYTE *)v7 == 1 )
    {
      v5 = 0;
      sub_CB59F0(a5, -*(_QWORD *)(v7 + 24));
    }
  }
  else if ( v6 > 110 )
  {
    if ( v6 == 115 && *(_BYTE *)v7 == 1 )
    {
      v5 = 0;
      sub_CB59F0(a5, -*(_DWORD *)(v7 + 24) & 0x1F);
    }
  }
  else
  {
    if ( v6 != 97 )
    {
      if ( v6 != 99 )
        return v5;
      v8 = *(_BYTE *)v7;
      goto LABEL_9;
    }
    v8 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 )
    {
LABEL_9:
      if ( v8 == 1 )
      {
        v5 = 0;
        sub_CB59F0(a5, *(_QWORD *)(v7 + 24));
      }
      else
      {
        v5 = 1;
        if ( v8 == 10 )
        {
          v11 = (void (__fastcall *)(__int64, __int64, __int64))*((_QWORD *)*a1 + 65);
          if ( v11 == sub_31F17D0 )
          {
            v5 = 0;
            v12 = sub_31DE680((__int64)a1, *(_QWORD *)(v7 + 24), (__int64)sub_31F17D0);
            sub_EA12C0(v12, a5, a1[26]);
            sub_31DCB40(
              (__int64)a1,
              *(unsigned int *)(v7 + 8) | (unsigned __int64)((__int64)*(int *)(v7 + 32) << 32),
              a5);
          }
          else
          {
            v5 = 0;
            v11((__int64)a1, v7, a5);
          }
        }
      }
      return v5;
    }
    v5 = 0;
    v10 = (__int64 (*)())*((_QWORD *)*a1 + 67);
    if ( v10 != sub_31F17A0 )
      ((void (__fastcall *)(_BYTE **, __int64, __int64, _QWORD))v10)(a1, a2, a3, 0);
  }
  return v5;
}
