// Function: sub_1157250
// Address: 0x1157250
//
__int64 __fastcall sub_1157250(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdx
  unsigned int v8; // esi

  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    a2,
    a3,
    a1[7],
    a1[8]);
  v5 = *a1;
  v6 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v6 )
  {
    do
    {
      v7 = *(_QWORD *)(v5 + 8);
      v8 = *(_DWORD *)v5;
      v5 += 16;
      sub_B99FD0(a2, v8, v7);
    }
    while ( v6 != v5 );
  }
  return a2;
}
