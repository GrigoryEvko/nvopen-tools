// Function: sub_250CD80
// Address: 0x250cd80
//
__int64 __fastcall sub_250CD80(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // rdx
  int v3; // eax
  unsigned __int64 v5; // rax
  __int64 v6; // rcx

  v2 = (unsigned __int8 *)(*a1 & 0xFFFFFFFFFFFFFFFCLL);
  if ( (*a1 & 3) == 3 )
    v2 = (unsigned __int8 *)*((_QWORD *)v2 + 3);
  v3 = *v2;
  if ( (unsigned __int8)v3 > 0x1Cu
    && (v5 = (unsigned int)(v3 - 34), (unsigned __int8)v5 <= 0x33u)
    && (v6 = 0x8000000000041LL, _bittest64(&v6, v5)) )
  {
    return *((_QWORD *)v2 + 9);
  }
  else
  {
    return *((_QWORD *)sub_250CBE0(a1, a2) + 15);
  }
}
