// Function: sub_1F987D0
// Address: 0x1f987d0
//
__int64 __fastcall sub_1F987D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 j; // rbx
  __int64 *v7; // rax
  int v8; // edx
  __int64 (__fastcall **i)(); // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+10h] [rbp-30h]
  __int64 *v13; // [rsp+18h] [rbp-28h]

  v2 = *a1;
  v13 = a1;
  v3 = *(_QWORD *)(v2 + 664);
  v12 = v2;
  v11 = v3;
  *(_QWORD *)(v2 + 664) = &i;
  v4 = *(_QWORD *)(a2 + 48);
  for ( i = off_49FFF30; v4; v4 = *(_QWORD *)(v4 + 32) )
    sub_1F81BC0((__int64)a1, *(_QWORD *)(v4 + 16));
  do
  {
    v5 = *(unsigned int *)(a2 + 56);
    if ( (_DWORD)v5 )
    {
      for ( j = 0; j != v5; ++j )
      {
        v7 = (__int64 *)(*(_QWORD *)(a2 + 32) + 40 * j);
        v8 = j;
        sub_1D44C70(*a1, a2, v8, *v7, v7[1]);
      }
    }
  }
  while ( *(_QWORD *)(a2 + 48) );
  sub_1F81E80(a1, a2);
  *(_QWORD *)(v12 + 664) = v11;
  return a2;
}
