// Function: sub_AA6260
// Address: 0xaa6260
//
__int64 __fastcall sub_AA6260(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  int v4; // esi
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // r8
  unsigned int v8; // edx
  int v9; // esi
  int v10; // edx
  int v11; // r9d

  result = *(_QWORD *)sub_AA48A0(a1);
  if ( (*(_BYTE *)(result + 3520) & 1) != 0 )
  {
    v3 = result + 3528;
    v4 = 3;
  }
  else
  {
    v9 = *(_DWORD *)(result + 3536);
    v3 = *(_QWORD *)(result + 3528);
    if ( !v9 )
      return result;
    v4 = v9 - 1;
  }
  v5 = v4 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v6 = (__int64 *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a1 == *v6 )
  {
LABEL_4:
    *v6 = -8192;
    v8 = *(_DWORD *)(result + 3520);
    ++*(_DWORD *)(result + 3524);
    *(_DWORD *)(result + 3520) = (2 * (v8 >> 1) - 2) | v8 & 1;
  }
  else
  {
    v10 = 1;
    while ( v7 != -4096 )
    {
      v11 = v10 + 1;
      v5 = v4 & (v10 + v5);
      v6 = (__int64 *)(v3 + 16LL * v5);
      v7 = *v6;
      if ( a1 == *v6 )
        goto LABEL_4;
      v10 = v11;
    }
  }
  return result;
}
