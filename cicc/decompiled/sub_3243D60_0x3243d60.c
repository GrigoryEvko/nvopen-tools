// Function: sub_3243D60
// Address: 0x3243d60
//
void __fastcall sub_3243D60(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rax
  __int64 v4; // [rsp-48h] [rbp-48h] BYREF
  unsigned __int64 v5; // [rsp-40h] [rbp-40h]
  char v6; // [rsp-38h] [rbp-38h]

  if ( a2 )
  {
    sub_AF47B0((__int64)&v4, *(unsigned __int64 **)(a2 + 16), *(unsigned __int64 **)(a2 + 24));
    if ( v6 )
    {
      sub_AF47B0((__int64)&v4, *(unsigned __int64 **)(a2 + 16), *(unsigned __int64 **)(a2 + 24));
      v2 = v5;
      v3 = a1[11];
      if ( v3 < v5 )
        sub_32422A0(a1, v5 - v3, 0);
      a1[11] = v2;
    }
  }
}
