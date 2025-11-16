// Function: sub_16AFE60
// Address: 0x16afe60
//
__int64 __fastcall sub_16AFE60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 result; // rax

  v3 = *(unsigned int *)(a2 + 8);
  if ( *(_DWORD *)(a2 + 8) )
  {
    v5 = 0;
    do
    {
      v6 = v5++;
      v7 = *(_QWORD *)(*(_QWORD *)a2 + 16 * v6 + 8);
      result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 48LL))(v7, a3);
    }
    while ( v5 != v3 );
  }
  return result;
}
