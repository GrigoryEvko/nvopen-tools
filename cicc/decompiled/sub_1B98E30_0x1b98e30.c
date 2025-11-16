// Function: sub_1B98E30
// Address: 0x1b98e30
//
__int64 __fastcall sub_1B98E30(__int64 a1, unsigned int a2, int a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  unsigned int v5; // eax
  _QWORD *v6; // rdx
  __int64 v7; // r15
  __int64 v8; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-38h]
  unsigned int v10; // [rsp+1Ch] [rbp-34h]

  result = a1 + 48;
  v4 = a3 + 1;
  if ( a2 < a3 + 1 )
  {
    do
    {
      while ( 1 )
      {
        v9 = a2;
        v10 = v4;
        sub_1B93930(&v8, (__int64 *)a1);
        v5 = *(_DWORD *)(a1 + 56);
        if ( v5 >= *(_DWORD *)(a1 + 60) )
        {
          sub_1B98CA0(a1 + 48, 0);
          v5 = *(_DWORD *)(a1 + 56);
        }
        v6 = (_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * v5);
        if ( !v6 )
          break;
        result = v8;
        *v6 = v8;
        ++*(_DWORD *)(a1 + 56);
LABEL_4:
        a2 = v10;
        if ( v10 >= v4 )
          return result;
      }
      v7 = v8;
      result = v5 + 1;
      *(_DWORD *)(a1 + 56) = result;
      if ( !v7 )
        goto LABEL_4;
      sub_1B949D0(v7);
      result = j_j___libc_free_0(v7, 472);
      a2 = v10;
    }
    while ( v10 < v4 );
  }
  return result;
}
