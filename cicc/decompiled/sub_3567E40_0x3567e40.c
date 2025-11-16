// Function: sub_3567E40
// Address: 0x3567e40
//
__int64 __fastcall sub_3567E40(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rdx
  __int64 result; // rax
  __int64 *v5; // r12
  __int64 *i; // r15
  __int64 v7; // r14
  __int64 *v8; // r12
  __int64 *j; // r14
  __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned __int64 v13; // [rsp+0h] [rbp-40h]

  if ( !(unsigned __int8)sub_3567D90(a1, a2) )
    sub_C64ED0("Broken region found: enumerated BB not in region!", 1u);
  v2 = a1[4];
  v13 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(__int64 **)(a2 + 112);
  result = *(unsigned int *)(a2 + 120);
  v5 = &v3[result];
  for ( i = v3; v5 != i; ++i )
  {
    v7 = *i;
    result = sub_3567D90(a1, *i);
    if ( v7 != v2 && (_BYTE)result != 1 )
      sub_C64ED0("Broken region found: edges leaving the region must go to the exit node!", 1u);
  }
  if ( v13 != a2 )
  {
    v8 = *(__int64 **)(a2 + 64);
    result = *(unsigned int *)(a2 + 72);
    for ( j = &v8[result]; j != v8; ++v8 )
    {
      v11 = *v8;
      result = sub_3567D90(a1, *v8);
      if ( !(_BYTE)result )
      {
        v12 = a1[3];
        if ( v11 )
        {
          v10 = (unsigned int)(*(_DWORD *)(v11 + 24) + 1);
          result = v10;
        }
        else
        {
          v10 = 0;
          result = 0;
        }
        if ( (unsigned int)result < *(_DWORD *)(v12 + 32) )
        {
          result = *(_QWORD *)(v12 + 24);
          if ( *(_QWORD *)(result + 8 * v10) )
            sub_C64ED0("Broken region found: edges entering the region must go to the entry node!", 1u);
        }
      }
    }
  }
  return result;
}
