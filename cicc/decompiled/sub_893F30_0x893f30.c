// Function: sub_893F30
// Address: 0x893f30
//
__int64 __fastcall sub_893F30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rbx
  char v7; // al
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi

  if ( a1 )
  {
    v6 = a1;
    while ( 1 )
    {
      v7 = *((_BYTE *)v6 + 8);
      switch ( v7 )
      {
        case 0:
          result = sub_8D97B0(v6[4]);
          goto LABEL_8;
        case 1:
          v9 = v6[6];
          if ( v9 )
          {
            result = sub_697870(v9);
            goto LABEL_8;
          }
          if ( (v6[3] & 1) == 0 )
          {
            v11 = v6[4];
            if ( v11 )
            {
              result = sub_7323F0(v11, a2, a3, a4, a5, a6);
LABEL_8:
              if ( (_DWORD)result )
                return result;
            }
          }
          break;
        case 2:
          v10 = v6[4];
          if ( v10 )
          {
            result = (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v10 + 88LL) + 160LL) & 4) != 0;
            goto LABEL_8;
          }
          break;
      }
      v6 = (__int64 *)*v6;
      if ( !v6 )
        return 0;
    }
  }
  return 0;
}
