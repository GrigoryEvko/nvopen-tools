// Function: sub_2845AB0
// Address: 0x2845ab0
//
__int64 __fastcall sub_2845AB0(__int64 a1)
{
  unsigned __int64 v1; // rax
  char v2; // dl
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  unsigned int v6; // r13d
  _QWORD *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rax

  v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 48 )
    goto LABEL_28;
  if ( !v1 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v1 - 24) - 30 > 0xA )
LABEL_28:
    BUG();
  v2 = *(_BYTE *)(v1 - 24);
  if ( v2 != 31 )
  {
    v4 = 0;
    if ( v2 == 32 )
    {
      v8 = *(_QWORD **)(v1 - 32);
      if ( *(_BYTE *)*v8 == 17 )
      {
        v9 = 0;
        v10 = ((*(_DWORD *)(v1 - 20) & 0x7FFFFFFu) >> 1) - 1;
        while ( 1 )
        {
          if ( v9 == v10 )
            return v8[4];
          v11 = v8[4 * (unsigned int)(2 * (v9 + 1))];
          if ( *v8 == v11 )
          {
            if ( v11 )
              break;
          }
          ++v9;
        }
        v12 = 4;
        if ( (_DWORD)v9 != -2 )
          v12 = 4LL * (unsigned int)(2 * v9 + 3);
        return v8[v12];
      }
    }
    return v4;
  }
  if ( (*(_DWORD *)(v1 - 20) & 0x7FFFFFF) != 1 )
  {
    v3 = *(_QWORD *)(v1 - 56);
    v4 = *(_QWORD *)(v1 - 88);
    if ( v3 == v4 )
      return v4;
    v5 = *(_QWORD *)(v1 - 120);
    if ( *(_BYTE *)v5 == 17 )
    {
      v6 = *(_DWORD *)(v5 + 32);
      if ( v6 <= 0x40 )
      {
        if ( *(_QWORD *)(v5 + 24) )
          return *(_QWORD *)(v1 - 56);
      }
      else if ( v6 != (unsigned int)sub_C444A0(v5 + 24) )
      {
        return v3;
      }
      return v4;
    }
  }
  return 0;
}
