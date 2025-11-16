// Function: sub_318DFC0
// Address: 0x318dfc0
//
__int64 __fastcall sub_318DFC0(__int64 a1)
{
  __int64 *v2; // rbx
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 v5; // rdi
  __int64 *v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rdi

  v2 = *(__int64 **)a1;
  result = *(unsigned int *)(a1 + 8);
  *(_DWORD *)(a1 + 64) = 0;
  v4 = &v2[result];
  if ( v4 != v2 )
  {
    do
    {
      v5 = *v2++;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
    }
    while ( v4 != v2 );
    v6 = *(__int64 **)a1;
    result = *(unsigned int *)(a1 + 8);
    v7 = *(_QWORD *)a1 + 8 * result;
    while ( v6 != (__int64 *)v7 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v7 - 8);
        v7 -= 8;
        if ( !v8 )
          break;
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 24LL))(v8);
        if ( v6 == (__int64 *)v7 )
          goto LABEL_7;
      }
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 8) = 0;
  return result;
}
