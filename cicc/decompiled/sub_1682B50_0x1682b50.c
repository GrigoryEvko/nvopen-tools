// Function: sub_1682B50
// Address: 0x1682b50
//
__int64 __fastcall sub_1682B50(
        __int64 (__fastcall *a1)(__int64, _QWORD *, _QWORD),
        __int64 a2,
        __int64 a3,
        _QWORD *a4,
        unsigned int a5)
{
  int *v8; // rbx
  int v9; // eax
  __int64 result; // rax
  int v11; // esi
  int v13; // [rsp+Ch] [rbp-44h]
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = __errno_location();
  v9 = *v8;
  *v8 = 0;
  v13 = v9;
  result = a1(a3, v14, a5);
  if ( v14[0] == a3 )
    sub_426290(a2);
  v11 = *v8;
  if ( *v8 == 34 || (unsigned __int64)(result + 0x80000000LL) > 0xFFFFFFFF )
    sub_426320(a2);
  if ( a4 )
  {
    *a4 = v14[0] - a3;
    v11 = *v8;
  }
  if ( !v11 )
    *v8 = v13;
  return result;
}
