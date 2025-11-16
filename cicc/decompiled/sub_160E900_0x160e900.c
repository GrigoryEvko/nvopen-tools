// Function: sub_160E900
// Address: 0x160e900
//
void __fastcall sub_160E900(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 *v6; // rbx
  __int64 *v7; // r12
  __int64 v8; // rdi
  __int64 v9; // rax

  if ( dword_4F9EB40 > 1 )
  {
    v2 = *(unsigned int *)(a1 + 264);
    if ( (_DWORD)v2 )
    {
      v3 = 8 * v2;
      v4 = 0;
      do
      {
        v5 = *(_QWORD *)(*(_QWORD *)(a1 + 256) + v4);
        v4 += 8;
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 136LL))(v5, 0);
      }
      while ( v3 != v4 );
    }
    v6 = *(__int64 **)(a1 + 32);
    v7 = &v6[*(unsigned int *)(a1 + 40)];
    while ( v7 != v6 )
    {
      v8 = *v6++;
      v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL))(v8);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 136LL))(v9, 1);
    }
  }
}
