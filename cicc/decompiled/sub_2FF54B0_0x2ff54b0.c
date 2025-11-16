// Function: sub_2FF54B0
// Address: 0x2ff54b0
//
__int64 __fastcall sub_2FF54B0(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 (*v3)(); // rax
  __int64 v5; // rax

  if ( a2
    && (v3 = *(__int64 (**)())(*(_QWORD *)a1 + 88LL), v3 != sub_2DCA420)
    && (v5 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v3)(
               a1,
               a3,
               (*(_WORD *)(*(_QWORD *)a3 + 2LL) >> 4) & 0x3FF)) != 0 )
  {
    return (*(_DWORD *)(v5 + 4LL * (a2 >> 5)) >> a2) & 1;
  }
  else
  {
    return 0;
  }
}
