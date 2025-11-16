// Function: sub_10E4DD0
// Address: 0x10e4dd0
//
__int64 __fastcall sub_10E4DD0(__int64 a1, _BYTE *a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rax
  char v4; // dl
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  unsigned int v8; // r14d
  _QWORD *v9; // rax
  __int64 v10; // rdx

  result = 0;
  if ( *a2 == 82 )
  {
    v3 = sub_B53900((__int64)a2);
    sub_B53630(v3, *(_QWORD *)a1);
    if ( !v4 )
      return 0;
    v5 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v5 != 57 )
      return 0;
    v6 = *((_QWORD *)v5 - 8);
    if ( !v6 )
      return 0;
    **(_QWORD **)(a1 + 8) = v6;
    v7 = *((_QWORD *)v5 - 4);
    if ( *(_BYTE *)v7 != 17 )
      return 0;
    v8 = *(_DWORD *)(v7 + 32);
    if ( v8 > 0x40 )
    {
      if ( v8 - (unsigned int)sub_C444A0(v7 + 24) > 0x40 )
        return 0;
      v9 = *(_QWORD **)(a1 + 16);
      v10 = **(_QWORD **)(v7 + 24);
    }
    else
    {
      v9 = *(_QWORD **)(a1 + 16);
      v10 = *(_QWORD *)(v7 + 24);
    }
    *v9 = v10;
    result = sub_10DF930(*((_QWORD *)a2 - 4));
    if ( (_BYTE)result )
      return result;
    return 0;
  }
  return result;
}
