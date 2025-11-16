// Function: sub_5E5D30
// Address: 0x5e5d30
//
__int64 __fastcall sub_5E5D30(__int64 a1, __int64 *a2, int a3)
{
  char *v3; // r15
  __int64 result; // rax
  __int64 v5; // r12
  int v6; // ebx
  __int64 v7; // rax
  int v8; // r15d
  __int16 v9; // r12
  __int64 v10; // rax

  v3 = "__device__";
  if ( !a3 )
    v3 = "__host__ __device__";
  result = sub_8D3410(a1);
  if ( (_DWORD)result )
  {
    v5 = a1;
    v6 = 0;
    if ( a1 )
    {
      while ( 1 )
      {
        result = sub_8D3410(v5);
        if ( !(_DWORD)result )
          break;
        ++v6;
        result = sub_8D4050(v5);
        v5 = result;
        if ( !result )
        {
          if ( v6 > 7 )
            result = sub_686310(7, 3597, a2, v3, a1);
          goto LABEL_5;
        }
      }
      if ( v6 > 7 )
        result = sub_686310(7, 3597, a2, v3, a1);
      while ( *(_BYTE *)(v5 + 140) == 12 )
        v5 = *(_QWORD *)(v5 + 160);
      if ( unk_4F04C44 != -1 )
        return result;
      result = unk_4F04C68 + 776LL * unk_4F04C64;
      if ( (*(_BYTE *)(result + 6) & 6) != 0 )
        return result;
      if ( *(_BYTE *)(result + 4) == 12 )
        goto LABEL_12;
      if ( !(unsigned int)sub_6CC470(30, v5, 0) )
        sub_686310(7, 3598, a2, v3, v5);
      v10 = sub_72D600(v5);
      result = sub_69A3A0(60, v10, v5);
      if ( !(_DWORD)result )
        result = sub_686310(7, 3599, a2, v3, v5);
    }
  }
LABEL_5:
  if ( unk_4F04C44 == -1 )
  {
    result = unk_4F04C68 + 776LL * unk_4F04C64;
    if ( (*(_BYTE *)(result + 6) & 6) == 0 )
    {
LABEL_12:
      if ( *(_BYTE *)(result + 4) != 12 )
      {
        v7 = *a2;
        dword_4CF7FB8 = 0;
        v8 = dword_4F07508[0];
        v9 = dword_4F07508[1];
        *(_QWORD *)dword_4F07508 = v7;
        result = sub_8D9600(a1, sub_5E4F90, 792);
        dword_4F07508[0] = v8;
        LOWORD(dword_4F07508[1]) = v9;
      }
    }
  }
  return result;
}
