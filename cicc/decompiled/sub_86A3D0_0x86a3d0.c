// Function: sub_86A3D0
// Address: 0x86a3d0
//
__int64 __fastcall sub_86A3D0(__int64 a1, __int64 a2, __int64 a3, char a4, const __m128i *a5)
{
  __int64 v5; // r12
  __int64 v7; // rax

  v5 = 0;
  if ( dword_4F04C3C )
    return v5;
  v7 = sub_86A320(a1, a2, a3, a4);
  v5 = v7;
  if ( !v7 || !a5 )
    return v5;
  *(_QWORD *)(v7 + 8) = sub_729420(*(_BYTE *)(v7 - 8) & 1, a5);
  return v5;
}
