// Function: sub_B6E8B0
// Address: 0xb6e8b0
//
__int64 __fastcall sub_B6E8B0(__int64 *a1, __int64 *a2, char a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7; // rdi
  __int64 result; // rax

  v4 = *a2;
  v5 = *a1;
  *a2 = 0;
  v7 = *(_QWORD *)(v5 + 104);
  *(_QWORD *)(v5 + 104) = v4;
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  result = *a1;
  *(_BYTE *)(*a1 + 112) = a3;
  return result;
}
