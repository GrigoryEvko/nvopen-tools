// Function: sub_2FF6320
// Address: 0x2ff6320
//
__int64 *__fastcall sub_2FF6320(__int64 *a1, int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 v8; // rax

  v8 = sub_22077B0(0x20u);
  if ( v8 )
  {
    *(_DWORD *)v8 = a2;
    *(_QWORD *)(v8 + 8) = a3;
    *(_DWORD *)(v8 + 16) = a4;
    *(_QWORD *)(v8 + 24) = a5;
  }
  *a1 = v8;
  a1[2] = (__int64)sub_2FF5600;
  a1[3] = (__int64)sub_2FF5BC0;
  return a1;
}
