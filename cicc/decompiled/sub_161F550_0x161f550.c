// Function: sub_161F550
// Address: 0x161f550
//
unsigned __int64 __fastcall sub_161F550(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // r12
  unsigned __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rsi

  v1 = *(__int64 **)(a1 + 56);
  v2 = *v1;
  result = *((unsigned int *)v1 + 2);
  v4 = *v1 + 8 * result;
  while ( v2 != v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 - 8);
      v4 -= 8;
      if ( !v5 )
        break;
      result = sub_161E7C0(v4, v5);
      if ( v2 == v4 )
        goto LABEL_5;
    }
  }
LABEL_5:
  *((_DWORD *)v1 + 2) = 0;
  return result;
}
