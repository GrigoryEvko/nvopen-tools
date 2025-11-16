// Function: sub_8645D0
// Address: 0x8645d0
//
unsigned int *__fastcall sub_8645D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  unsigned int *result; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rsi
  bool v11; // zf
  int v12; // edi
  __int64 v13; // rax
  int v14; // esi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r13

  result = (unsigned int *)dword_4F04C64;
  v7 = qword_4F04C68[0];
  v8 = 776LL * dword_4F04C64;
  v9 = qword_4F04C68[0] + v8;
  v10 = *(_QWORD *)(qword_4F04C68[0] + v8 + 528);
  if ( v10 > 0 )
  {
    v11 = *(_BYTE *)(v9 + 4) == 8;
    v12 = dword_4F04C64;
    *(_QWORD *)(v9 + 528) = v10 - 1;
    if ( !v11 )
      return result;
    goto LABEL_8;
  }
  v13 = *(_QWORD *)(*(_QWORD *)(v9 + 224) + 40LL);
  if ( v13 && *(_BYTE *)(v13 + 28) == 3 )
  {
    v17 = *(_QWORD *)(v13 + 32);
    sub_863FC0(a1, v10, v9, qword_4F04C68[0], v8, a6);
    if ( v17 )
      sub_8645D0();
  }
  else
  {
    sub_863FC0(a1, v10, v9, qword_4F04C68[0], v8, a6);
  }
  result = (unsigned int *)dword_4F04C64;
  v7 = qword_4F04C68[0];
  v8 = 776LL * dword_4F04C64;
  v12 = dword_4F04C64;
  v9 = qword_4F04C68[0] + v8;
  if ( *(_BYTE *)(qword_4F04C68[0] + v8 + 4) == 8 )
  {
LABEL_8:
    if ( *(_BYTE *)(v7 + v8 - 772) == 8 )
    {
      v14 = v12;
      v15 = 776LL * (_QWORD)result - 776;
      do
      {
        v16 = v15;
        v15 -= 776;
        --v14;
      }
      while ( *(_BYTE *)(v7 + v15 + 4) == 8 );
      v9 = v7 + v16;
    }
    else
    {
      v14 = v12;
    }
    *(_DWORD *)(v9 + 552) = v14 - 1;
    dword_4F04C60 = v12;
    return &dword_4F04C60;
  }
  return result;
}
