// Function: sub_1210D90
// Address: 0x1210d90
//
__int64 __fastcall sub_1210D90(__int64 a1)
{
  int v1; // eax
  int v2; // r13d
  int v3; // edx
  unsigned int v4; // r13d
  int v6; // eax
  int v7; // ebx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rsi
  _QWORD v10[4]; // [rsp+0h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-30h]
  char v12; // [rsp+21h] [rbp-2Fh]

  v3 = *(_DWORD *)(a1 + 240);
  if ( v3 == 411 )
  {
LABEL_6:
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
    if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' at start of summary entry") )
      return 1;
    v4 = sub_120AFE0(a1, 12, "expected '(' at start of summary entry");
    if ( (_BYTE)v4 )
    {
      return 1;
    }
    else
    {
      v6 = *(_DWORD *)(a1 + 240);
      v7 = 1;
      do
      {
        if ( v6 == 12 )
        {
LABEL_13:
          ++v7;
        }
        else
        {
          while ( v6 != 13 )
          {
            if ( !v6 )
            {
              v12 = 1;
              v8 = *(_QWORD *)(a1 + 232);
              v11 = 3;
              v10[0] = "found end of file while parsing summary entry";
              sub_11FD800(a1 + 176, v8, (__int64)v10, 1);
              return 1;
            }
            v6 = sub_1205200(a1 + 176);
            *(_DWORD *)(a1 + 240) = v6;
            if ( v6 == 12 )
              goto LABEL_13;
          }
          --v7;
        }
        v6 = sub_1205200(a1 + 176);
        *(_DWORD *)(a1 + 240) = v6;
      }
      while ( v7 );
    }
    return v4;
  }
  LOBYTE(v2) = v3 != 461;
  LOBYTE(v1) = v3 != 100;
  v4 = v1 & v2;
  LOBYTE(v4) = ((unsigned int)(v3 - 415) > 1) & v4;
  if ( (_BYTE)v4 )
  {
    v9 = *(_QWORD *)(a1 + 232);
    v12 = 1;
    v10[0] = "Expected 'gv', 'module', 'typeid', 'flags' or 'blockcount' at the start of summary entry";
    v11 = 3;
    sub_11FD800(a1 + 176, v9, (__int64)v10, 1);
    return v4;
  }
  if ( v3 != 415 )
  {
    if ( v3 == 416 )
      return sub_1210D20(a1);
    goto LABEL_6;
  }
  return sub_1210CA0(a1);
}
