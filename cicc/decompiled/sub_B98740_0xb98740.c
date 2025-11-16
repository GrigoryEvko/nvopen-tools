// Function: sub_B98740
// Address: 0xb98740
//
void __fastcall sub_B98740(_WORD *a1, unsigned __int64 a2)
{
  int v2; // r13d
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rsi

  v2 = a2;
  if ( (*(_BYTE *)a1 & 2) != 0 )
  {
    v3 = *((unsigned int *)a1 - 2);
    v4 = *((_QWORD *)a1 - 2);
    if ( a2 != v3 )
    {
      v5 = 8 * a2;
      if ( a2 < v3 )
      {
        v8 = v4 + 8 * v3;
        v9 = v4 + v5;
        while ( v9 != v8 )
        {
          v10 = *(_QWORD *)(v8 - 8);
          v8 -= 8;
          if ( v10 )
            sub_B91220(v8, v10);
        }
      }
      else
      {
        if ( a2 > *((unsigned int *)a1 - 1) )
        {
          sub_B97700((__int64)(a1 - 8), a2);
          v4 = *((_QWORD *)a1 - 2);
          v3 = *((unsigned int *)a1 - 2);
        }
        v6 = (_QWORD *)(v4 + 8 * v3);
        for ( i = (_QWORD *)(v5 + v4); i != v6; ++v6 )
        {
          if ( v6 )
            *v6 = 0;
        }
      }
      *((_DWORD *)a1 - 2) = v2;
    }
  }
  else if ( a2 != ((*a1 >> 6) & 0xF) )
  {
    if ( a2 > ((*(_BYTE *)a1 >> 2) & 0xFu) )
      sub_B98540(a1, a2);
    else
      sub_B91520(a1, a2);
  }
}
