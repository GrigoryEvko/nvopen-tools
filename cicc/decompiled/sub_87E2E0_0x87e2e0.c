// Function: sub_87E2E0
// Address: 0x87e2e0
//
_QWORD *__fastcall sub_87E2E0(__int64 a1, int a2, __int64 *a3, __int64 a4)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rax
  _QWORD *result; // rax
  _QWORD *v10; // rcx

  v6 = (_QWORD *)sub_823970(32);
  *v6 = 0;
  v7 = v6;
  v8 = *a3;
  *((_DWORD *)v7 + 4) = a2;
  v7[1] = v8;
  v7[3] = a4;
  result = *(_QWORD **)(a1 + 32);
  if ( result )
  {
    do
    {
      v10 = result;
      result = (_QWORD *)*result;
    }
    while ( result );
    *v10 = v7;
  }
  else
  {
    *(_QWORD *)(a1 + 32) = v7;
  }
  return result;
}
