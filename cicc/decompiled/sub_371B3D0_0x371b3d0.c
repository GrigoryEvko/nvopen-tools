// Function: sub_371B3D0
// Address: 0x371b3d0
//
__int64 __fastcall sub_371B3D0(__int64 a1)
{
  _QWORD *v1; // rsi
  __int64 v2; // rax
  unsigned int v3; // eax
  char v4; // si
  char v5; // bl
  __int64 v6; // rcx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int16 v10; // dx

  v1 = *(_QWORD **)(a1 + 8);
  if ( v1 == (_QWORD *)(*(_QWORD *)a1 + 48LL) )
  {
    *(_QWORD *)(a1 + 8) = *v1 & 0xFFFFFFFFFFFFFFF8LL;
    *(_WORD *)(a1 + 16) = 0;
    return a1;
  }
  else
  {
    v2 = sub_371B3B0(a1, (__int64)v1);
    v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 88LL))(v2);
    v4 = *(_BYTE *)(a1 + 16);
    v5 = *(_BYTE *)(a1 + 17);
    v6 = v3;
    v7 = *(_QWORD **)(a1 + 8);
    v8 = 1 - v6;
    if ( v6 )
    {
      do
      {
        ++v8;
        v7 = (_QWORD *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
      }
      while ( v8 != 1 );
      v5 = 0;
      v4 = 0;
    }
    *(_QWORD *)(a1 + 8) = v7;
    LOBYTE(v10) = v4;
    HIBYTE(v10) = v5;
    *(_WORD *)(a1 + 16) = v10;
    return a1;
  }
}
