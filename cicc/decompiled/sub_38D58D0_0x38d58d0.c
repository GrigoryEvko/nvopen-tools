// Function: sub_38D58D0
// Address: 0x38d58d0
//
__int64 __fastcall sub_38D58D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r14d
  __int64 (__fastcall *v6)(__int64); // rax
  __int64 v7; // rdx
  unsigned __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_38D4150(a1, 0, 0);
  sub_38BE3C0(*(_QWORD *)(a1 + 8));
  *(_BYTE *)(*(_QWORD *)(a1 + 8) + 1040LL) = 0;
  v4 = sub_390D400(*(_QWORD *)(a1 + 264), a2);
  v9[0] = 0;
  v5 = v4;
  if ( a3 )
  {
    v6 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL);
    if ( v6 == sub_38D3BD0 )
    {
      LODWORD(v7) = 0;
      if ( *(_BYTE *)(a1 + 260) )
        v7 = *(_QWORD *)(a1 + 264);
    }
    else
    {
      LODWORD(v7) = v6(a1);
    }
    if ( !sub_38CF2B0(a3, v9, v7) )
      sub_16BD130("Cannot evaluate subsection number", 1u);
    if ( v9[0] > 0x2000 )
      sub_16BD130("Subsection number out of range", 1u);
  }
  *(_QWORD *)(a1 + 272) = sub_38D78D0(a2);
  return v5;
}
