// Function: sub_ECD7F0
// Address: 0xecd7f0
//
__int64 __fastcall sub_ECD7F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rsi

  if ( *(_DWORD *)sub_ECD7B0(a1) == 1 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
  v3 = *(_QWORD *)(a1 + 16);
  v4 = v3 + 112LL * *(unsigned int *)(a1 + 24);
  while ( v3 != v4 )
  {
    v5 = v3 + 8;
    v3 += 112;
    sub_CA0EC0(a2, v5);
  }
  return 1;
}
