// Function: sub_20EA230
// Address: 0x20ea230
//
__int64 __fastcall sub_20EA230(__int64 a1, int a2, int a3)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  int v5; // eax

  v3 = *(_QWORD *)(a1 + 56);
  v4 = a3 & 0x7FFFFFFF;
  v5 = *(_DWORD *)(*(_QWORD *)(v3 + 264) + 4 * v4);
  if ( v5 )
    return sub_1F5BDB0(v3, a2, v5);
  else
    return sub_1F5BF40(v3, a2, *(_DWORD *)(*(_QWORD *)(v3 + 288) + 4 * v4));
}
