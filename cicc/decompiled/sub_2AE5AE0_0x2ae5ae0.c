// Function: sub_2AE5AE0
// Address: 0x2ae5ae0
//
__int64 *__fastcall sub_2AE5AE0(__int64 a1, unsigned __int64 a2)
{
  int v2; // edx
  __int64 *result; // rax
  int v4; // ecx
  __int64 v5; // r9
  int v6; // esi
  int v7; // ecx
  int v8; // r10d
  unsigned int i; // esi
  __int64 v10; // r8
  unsigned int v11; // esi

  v2 = a2;
  result = (__int64 *)HIDWORD(a2);
  if ( !BYTE4(a2) )
  {
    if ( (_DWORD)a2 == 1 )
      return result;
    v4 = *(_DWORD *)(a1 + 184);
    v5 = *(_QWORD *)(a1 + 168);
    if ( v4 )
    {
      v6 = 37 * a2;
      goto LABEL_5;
    }
LABEL_12:
    sub_2ACAC50(a1, a2);
    sub_2AE4570(a1, a2);
    sub_2AC7F80(a1, a2);
    return sub_2ADE2D0(a1, a2);
  }
  v4 = *(_DWORD *)(a1 + 184);
  v5 = *(_QWORD *)(a1 + 168);
  if ( !v4 )
    goto LABEL_12;
  v6 = 37 * a2 - 1;
LABEL_5:
  v7 = v4 - 1;
  v8 = 1;
  for ( i = v7 & v6; ; i = v7 & v11 )
  {
    v10 = v5 + 72LL * i;
    if ( *(_DWORD *)v10 == v2 && (_BYTE)result == *(_BYTE *)(v10 + 4) )
      break;
    if ( *(_DWORD *)v10 == -1 && *(_BYTE *)(v10 + 4) )
      goto LABEL_12;
    v11 = v8 + i;
    ++v8;
  }
  return result;
}
