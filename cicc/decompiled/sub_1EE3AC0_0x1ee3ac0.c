// Function: sub_1EE3AC0
// Address: 0x1ee3ac0
//
__int64 __fastcall sub_1EE3AC0(__int64 a1)
{
  __int64 v1; // r12
  int v2; // r8d
  int v3; // r9d
  __int64 result; // rax
  __int64 v5; // r14
  _QWORD *v6; // r13

  v1 = 0;
  sub_1EE1050(a1, *(__int64 **)(a1 + 480), *(_DWORD *)(a1 + 488));
  result = *(unsigned int *)(a1 + 488);
  v5 = 8 * result;
  if ( (_DWORD)result )
  {
    do
    {
      v6 = (_QWORD *)(v1 + *(_QWORD *)(a1 + 480));
      if ( *v6 )
      {
        result = *(unsigned int *)(a1 + 408);
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 412) )
        {
          sub_16CD150(a1 + 400, (const void *)(a1 + 416), 0, 8, v2, v3);
          result = *(unsigned int *)(a1 + 408);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * result) = *v6;
        ++*(_DWORD *)(a1 + 408);
      }
      v1 += 8;
    }
    while ( v5 != v1 );
  }
  *(_DWORD *)(a1 + 488) = 0;
  return result;
}
