// Function: sub_74DB20
// Address: 0x74db20
//
__int64 __fastcall sub_74DB20(__int64 a1, int a2, _QWORD *a3)
{
  __int64 (__fastcall *v3)(char *, _QWORD *); // rax

  v3 = (__int64 (__fastcall *)(char *, _QWORD *))*a3;
  if ( a2 )
  {
    v3("reinterpret_cast<", a3);
    sub_74B930(a1, (__int64)a3);
    return ((__int64 (__fastcall *)(const char *, _QWORD *))*a3)(">(", a3);
  }
  else
  {
    v3("(", a3);
    sub_74B930(a1, (__int64)a3);
    return ((__int64 (__fastcall *)(char *, _QWORD *))*a3)(")", a3);
  }
}
