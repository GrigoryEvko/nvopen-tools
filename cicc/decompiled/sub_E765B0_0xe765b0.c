// Function: sub_E765B0
// Address: 0xe765b0
//
_QWORD *__fastcall sub_E765B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  _DWORD *v8; // rsi
  __int64 v9; // rbx
  _BYTE *v11; // rdx
  _BYTE *v12; // rax

  v6 = a2 + 112;
  if ( !*(_BYTE *)(a2 + 157) )
    sub_C0D2A0(a2 + 112);
  a1[1] = 0;
  v8 = a1 + 3;
  *a1 = a1 + 3;
  a1[2] = 0;
  v9 = *(_QWORD *)(a2 + 144);
  if ( v9 )
  {
    sub_C8D290((__int64)a1, v8, v9, 1u, a5, a6);
    v8 = (_DWORD *)*a1;
    v11 = (_BYTE *)(*a1 + v9);
    v12 = (_BYTE *)(*a1 + a1[1]);
    if ( v12 != v11 )
    {
      do
      {
        if ( v12 )
          *v12 = 0;
        ++v12;
      }
      while ( v11 != v12 );
      v8 = (_DWORD *)*a1;
    }
    a1[1] = v9;
  }
  sub_C0BFF0(v6, v8);
  return a1;
}
