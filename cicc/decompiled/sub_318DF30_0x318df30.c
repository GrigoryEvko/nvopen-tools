// Function: sub_318DF30
// Address: 0x318df30
//
__int64 __fastcall sub_318DF30(__int64 *a1)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rbx
  void (__fastcall ***v5)(_QWORD, __int64 *); // rdi
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rdi

  v2 = *a1;
  result = *((unsigned int *)a1 + 2);
  *((_DWORD *)a1 + 16) = 2;
  v4 = v2 + 8 * result;
  if ( v2 != v4 )
  {
    do
    {
      v5 = *(void (__fastcall ****)(_QWORD, __int64 *))(v4 - 8);
      v4 -= 8;
      (**v5)(v5, a1);
    }
    while ( v2 != v4 );
    v6 = *a1;
    result = *((unsigned int *)a1 + 2);
    v7 = *a1 + 8 * result;
    while ( v6 != v7 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v7 - 8);
        v7 -= 8;
        if ( !v8 )
          break;
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 24LL))(v8);
        if ( v6 == v7 )
          goto LABEL_7;
      }
    }
  }
LABEL_7:
  *((_DWORD *)a1 + 2) = 0;
  *((_DWORD *)a1 + 16) = 0;
  return result;
}
