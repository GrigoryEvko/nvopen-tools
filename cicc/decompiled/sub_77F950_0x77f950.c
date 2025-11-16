// Function: sub_77F950
// Address: 0x77f950
//
unsigned __int64 __fastcall sub_77F950(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx

  result = (unsigned int)dword_4F08058;
  if ( !dword_4F08058 )
  {
    v5 = (_QWORD *)qword_4F083B0;
    if ( qword_4F083B0 )
    {
      do
      {
        v6 = v5;
        v5 = (_QWORD *)*v5;
        sub_822B90(v6, *((unsigned int *)v6 + 2), a3, a4);
      }
      while ( v5 );
    }
    v7 = qword_4F083A8;
    qword_4F083B0 = 0;
    if ( qword_4F083A8 )
    {
      if ( qword_4F082A0 )
      {
        v8 = qword_4F083A8;
        do
        {
          v9 = v8;
          v8 = *(_QWORD *)(v8 + 8);
        }
        while ( v8 );
        *(_QWORD *)(v9 + 8) = qword_4F082A0;
      }
      qword_4F082A0 = v7;
      qword_4F083A8 = 0;
    }
    result = (unsigned __int64)sub_770440((__int64)&qword_4F08380);
    qword_4F082A0 = 0;
  }
  return result;
}
