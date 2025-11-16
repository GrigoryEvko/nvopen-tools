// Function: sub_16BDCA0
// Address: 0x16bdca0
//
__int64 __fastcall sub_16BDCA0(__int64 a1, _QWORD *a2, unsigned int a3)
{
  _QWORD *v3; // r8
  unsigned __int64 v4; // rax
  _QWORD *v5; // rcx
  _QWORD *v6; // rax

  v3 = (_QWORD *)*a2;
  if ( *a2 )
  {
    --*(_DWORD *)(a1 + 20);
    v4 = (unsigned __int64)v3;
    *a2 = 0;
    while ( 1 )
    {
      if ( ((unsigned __int8)~(_BYTE)v4 & (v4 != 0)) != 0 )
      {
        v5 = *(_QWORD **)v4;
        if ( a2 == *(_QWORD **)v4 )
        {
          *(_QWORD *)v4 = v3;
          LOBYTE(a3) = ~(_BYTE)v4 & (v4 != 0);
          return a3;
        }
      }
      else
      {
        v6 = (_QWORD *)(v4 & 0xFFFFFFFFFFFFFFFELL);
        v5 = (_QWORD *)*v6;
        if ( a2 == (_QWORD *)*v6 )
        {
          *v6 = v3;
          return 1;
        }
      }
      v4 = (unsigned __int64)v5;
    }
  }
  return 0;
}
