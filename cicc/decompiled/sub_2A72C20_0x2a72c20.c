// Function: sub_2A72C20
// Address: 0x2a72c20
//
__int64 __fastcall sub_2A72C20(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 result; // rax
  __int64 v10; // r12
  char v11; // r15

  v6 = a2;
  v7 = *a1;
  do
  {
    sub_2A72020(v7, a2, a3, a4, a5, a6);
    v8 = *(__int64 **)v6;
    result = *(unsigned int *)(v6 + 8);
    v10 = *(_QWORD *)v6 + 8 * result;
    if ( *(_QWORD *)v6 == v10 )
      break;
    v11 = 0;
    do
    {
      a2 = *v8++;
      result = sub_2A6CD80(v7, a2, a3, a4);
      v11 |= result;
    }
    while ( (__int64 *)v10 != v8 );
  }
  while ( v11 );
  return result;
}
