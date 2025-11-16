// Function: sub_2EAA1A0
// Address: 0x2eaa1a0
//
__int64 __fastcall sub_2EAA1A0(__int64 a1)
{
  void (*v1)(); // rdx
  __int64 v2; // rcx
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  __int64 v5; // rsi
  _QWORD *v6; // r12
  unsigned __int64 v7; // r14

  sub_2EAA010(a1);
  v3 = *(unsigned int *)(a1 + 2528);
  if ( (_DWORD)v3 )
  {
    v4 = *(_QWORD **)(a1 + 2512);
    v5 = 2 * v3;
    v6 = &v4[v5];
    do
    {
      if ( *v4 != -8192 && *v4 != -4096 )
      {
        v7 = v4[1];
        if ( v7 )
        {
          sub_2E81F20(v4[1], v5 * 8, v1, v2);
          v5 = 140;
          j_j___libc_free_0(v7);
        }
      }
      v4 += 2;
    }
    while ( v6 != v4 );
    v3 = *(unsigned int *)(a1 + 2528);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2512), 16 * v3, 8);
  return sub_E68A10(a1 + 8);
}
