// Function: sub_C66990
// Address: 0xc66990
//
void __fastcall sub_C66990(__int64 a1, unsigned __int8 *a2, size_t a3)
{
  unsigned __int8 *v3; // r12
  unsigned __int64 v4; // r8

  if ( !*(_BYTE *)(a1 + 104) )
  {
    v3 = &a2[a3];
    v4 = *(_QWORD *)(a1 + 64);
    if ( (unsigned __int64)a2 <= v4 && v4 <= (unsigned __int64)v3 )
    {
      sub_C666F0(a1, *(unsigned __int8 **)(a1 + 64), a3 - (v4 - (_QWORD)a2));
      *(_QWORD *)(a1 + 64) = v3;
    }
    else
    {
      sub_C666F0(a1, a2, a3);
      *(_QWORD *)(a1 + 64) = v3;
    }
  }
}
