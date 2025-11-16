// Function: sub_2D00E90
// Address: 0x2d00e90
//
__int64 __fastcall sub_2D00E90(unsigned int *a1, unsigned __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // r12d
  __int64 v4; // r13
  __int64 v5; // rbx
  unsigned int v6; // eax
  int v7; // eax
  int v8; // r8d
  __int64 result; // rax

  v2 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v3 = sub_2D00C30(a1, **(_BYTE ***)(a2 - 8));
  if ( v2 > 1 )
  {
    v4 = 32LL * v2;
    v5 = 32;
    while ( 1 )
    {
      v6 = sub_2D00C30(a1, *(_BYTE **)(*(_QWORD *)(a2 - 8) + v5));
      v7 = sub_2D00850((__int64)a1, v6, v3);
      v3 = v7;
      if ( a1[1] == v7 )
        break;
      v5 += 32;
      if ( v5 == v4 )
        goto LABEL_6;
    }
    sub_2D00AD0(a1, a2, v7);
  }
LABEL_6:
  v8 = sub_2D00C30(a1, (_BYTE *)a2);
  result = 0;
  if ( v8 != v3 )
  {
    sub_2D00AD0(a1, a2, v3);
    return 1;
  }
  return result;
}
