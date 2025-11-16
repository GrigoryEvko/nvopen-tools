// Function: sub_388AF70
// Address: 0x388af70
//
__int64 __fastcall sub_388AF70(__int64 a1)
{
  __int64 v1; // r14
  int v2; // eax
  unsigned int v3; // r13d
  int v5; // eax
  int v6; // r12d
  const char *v7; // rax
  unsigned __int64 v8; // rsi
  const char *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  v1 = a1 + 8;
  v2 = *(_DWORD *)(a1 + 64);
  if ( v2 != 305 && v2 != 93 && v2 != 340 )
  {
    v11 = 1;
    v7 = "Expected 'gv', 'module', or 'typeid' at the start of summary entry";
LABEL_16:
    v8 = *(_QWORD *)(a1 + 56);
    v9 = v7;
    v10 = 3;
    return (unsigned int)sub_38814C0(v1, v8, (__int64)&v9);
  }
  else
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' at start of summary entry") )
      return 1;
    v3 = sub_388AF10(a1, 12, "expected '(' at start of summary entry");
    if ( (_BYTE)v3 )
    {
      return 1;
    }
    else
    {
      v5 = *(_DWORD *)(a1 + 64);
      v6 = 1;
      do
      {
        if ( v5 == 12 )
        {
LABEL_12:
          ++v6;
        }
        else
        {
          while ( v5 != 13 )
          {
            if ( !v5 )
            {
              v11 = 1;
              v7 = "found end of file while parsing summary entry";
              goto LABEL_16;
            }
            v5 = sub_3887100(v1);
            *(_DWORD *)(a1 + 64) = v5;
            if ( v5 == 12 )
              goto LABEL_12;
          }
          --v6;
        }
        v5 = sub_3887100(v1);
        *(_DWORD *)(a1 + 64) = v5;
      }
      while ( v6 );
    }
    return v3;
  }
}
