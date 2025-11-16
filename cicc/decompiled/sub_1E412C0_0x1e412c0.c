// Function: sub_1E412C0
// Address: 0x1e412c0
//
_QWORD *__fastcall sub_1E412C0(__int64 a1, __int64 a2, int a3, int a4)
{
  __int64 v4; // r8
  _BYTE *v5; // r9
  _QWORD *v6; // r12
  __int64 v8; // rcx
  __int64 v9; // r13
  __int64 i; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // eax

  v6 = sub_1E0B7C0(*(_QWORD *)(a1 + 32), a2);
  if ( **(_WORD **)(a2 + 16) == 1 )
  {
    v8 = *(unsigned int *)(a2 + 40);
    if ( (_DWORD)v8 )
    {
      v9 = (unsigned int)v8;
      for ( i = 0; i != v9; ++i )
      {
        v11 = *(_QWORD *)(a2 + 32);
        v12 = v11 + 40 * i;
        if ( !*(_BYTE *)v12 )
        {
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
            break;
          if ( (*(_WORD *)(v12 + 2) & 0xFF0) != 0 )
          {
            v13 = sub_1E16AB0(a2, i, v11, v8, v4, v5);
            sub_1E16A40((__int64)v6, i, v13);
          }
        }
      }
    }
  }
  if ( a3 != a4 )
    sub_1E41150(a1, (__int64)v6, a2, a3 - a4);
  return v6;
}
