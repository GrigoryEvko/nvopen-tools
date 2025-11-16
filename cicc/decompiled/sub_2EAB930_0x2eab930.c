// Function: sub_2EAB930
// Address: 0x2eab930
//
__int64 __fastcall sub_2EAB930(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  __int64 v3; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 && (v2 = *(_QWORD *)(v1 + 24)) != 0 && (v3 = *(_QWORD *)(v2 + 32)) != 0 )
    return sub_2EAAD50(*(_QWORD *)(v3 + 16), *(_DWORD *)(a1 + 24));
  else
    return 0;
}
