// Function: sub_7A8650
// Address: 0x7a8650
//
__int64 __fastcall sub_7A8650(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned int v3; // r9d
  __int64 v6; // rbx
  unsigned int v7; // r13d
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rdx

  v3 = 1;
  v6 = *(_QWORD *)(a2 + 40);
  if ( dword_4D0425C )
    v3 = ((*(_BYTE *)(a2 + 96) >> 1) ^ 1) & 1;
  v7 = sub_7A8210(a1, *(_QWORD *)(a2 + 40), a3, 0, 0, v3);
  if ( !v7 )
  {
    if ( dword_4D0425C && a3 && (*(_BYTE *)(a2 + 96) & 2) != 0 && !(unsigned int)sub_7A80B0(*(_QWORD *)(a2 + 40)) )
    {
      v12 = *(_QWORD **)(a2 + 112);
      while ( 1 )
      {
        v13 = v12[2];
        if ( v12[1] != v13 && a2 == *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 8) + 16LL) + 24LL) )
          break;
        v12 = (_QWORD *)*v12;
        if ( !v12 )
          return v7;
      }
    }
    v8 = *(_QWORD *)(a2 + 24);
    if ( !v8 || !(unsigned int)sub_7A8650(a1, v8, a3) )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(v6 + 168) + 8LL);
      if ( !v9 )
        return v7;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v9 + 96) & 2) == 0 )
        {
          v10 = sub_8E5650(v9);
          if ( *(_QWORD *)(a2 + 24) != v10 )
          {
            if ( (unsigned int)sub_7A8650(a1, v10, a3 + *(_QWORD *)(v9 + 104)) )
              break;
          }
        }
        v9 = *(_QWORD *)(v9 + 8);
        if ( !v9 )
          return v7;
      }
    }
  }
  return 1;
}
