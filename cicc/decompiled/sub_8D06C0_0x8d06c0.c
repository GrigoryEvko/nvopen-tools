// Function: sub_8D06C0
// Address: 0x8d06c0
//
__int64 __fastcall sub_8D06C0(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 i; // r13
  void *v4; // rcx
  __int64 v5; // rax
  int v6; // edi
  __int64 result; // rax
  __int64 v8; // rdx

  v2 = qword_4F60540;
  for ( i = a1[2]; v2; v2 = *(_QWORD *)v2 )
  {
    v4 = memcpy((void *)(i + *(_QWORD *)(v2 + 24)), *(const void **)(v2 + 8), *(_QWORD *)(v2 + 16));
    v5 = *(_QWORD *)(v2 + 32);
    if ( v5 )
      *(_QWORD *)((char *)a1 + v5) = v4;
  }
  v6 = dword_4F04C64;
  a1[25] = unk_4F07290;
  result = qword_4F07300;
  a1[31] = qword_4F072C0;
  a1[39] = result;
  if ( v6 != -1 )
  {
    sub_85FE80(v6, 0, 0);
    result = qword_4F04C68[0] + 776LL * dword_4F04C64;
    if ( result )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(result + 184);
        if ( v8 )
          *(_DWORD *)(v8 + 240) = -1;
        if ( !*(_BYTE *)(result + 4) )
          break;
        result -= 776;
      }
    }
  }
  return result;
}
