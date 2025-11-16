// Function: sub_39A7A30
// Address: 0x39a7a30
//
__int64 __fastcall sub_39A7A30(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rbx
  _BYTE *v8; // rdx

  sub_39A6760(a1, a2, *(_QWORD *)(a3 + 8 * (3LL - *(unsigned int *)(a3 + 8))), 73);
  result = 4LL - *(unsigned int *)(a3 + 8);
  v5 = *(_QWORD *)(a3 + 8 * result);
  if ( v5 )
  {
    result = *(unsigned int *)(v5 + 8);
    if ( (_DWORD)result )
    {
      v6 = *(unsigned int *)(v5 + 8);
      v7 = 0;
      while ( 1 )
      {
        v8 = *(_BYTE **)(v5 + 8 * (v7 - result));
        if ( *v8 == 34 )
          result = sub_39A5DF0(a1, a2, (__int64)v8);
        if ( v6 == ++v7 )
          break;
        result = *(unsigned int *)(v5 + 8);
      }
    }
  }
  return result;
}
