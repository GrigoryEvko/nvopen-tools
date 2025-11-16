// Function: sub_1212200
// Address: 0x1212200
//
__int64 __fastcall sub_1212200(__int64 a1, __m128i *a2)
{
  unsigned int v4; // r8d
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx

  if ( (unsigned __int8)sub_120AFE0(a1, 100, "expected 'module' here") )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    return 1;
  v4 = sub_120AFE0(a1, 506, "expected module ID");
  if ( (_BYTE)v4 )
    return 1;
  v5 = *(_QWORD *)(a1 + 1712);
  v6 = *(_DWORD *)(a1 + 280);
  v7 = a1 + 1704;
  if ( v5 )
  {
    v8 = a1 + 1704;
    do
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(v5 + 16);
        v10 = *(_QWORD *)(v5 + 24);
        if ( v6 <= *(_DWORD *)(v5 + 32) )
          break;
        v5 = *(_QWORD *)(v5 + 24);
        if ( !v10 )
          goto LABEL_10;
      }
      v8 = v5;
      v5 = *(_QWORD *)(v5 + 16);
    }
    while ( v9 );
LABEL_10:
    if ( v8 != v7 && v6 >= *(_DWORD *)(v8 + 32) )
      v7 = v8;
  }
  *a2 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  return v4;
}
