// Function: sub_30A68F0
// Address: 0x30a68f0
//
__int64 __fastcall sub_30A68F0(__int64 *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  __int64 v4; // rcx
  __int64 v5; // rdi
  unsigned int v6; // edx
  __int64 v7; // r10
  unsigned int v8; // r11d

  result = *a1;
  v3 = a2[2];
  v4 = *(unsigned int *)(*a1 + 24);
  v5 = *(_QWORD *)(*a1 + 8);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((0xBF58476D1CE4E5B9LL * v3) >> 31) ^ (484763065 * v3));
    result = v5 + 16LL * v6;
    v7 = *(_QWORD *)result;
    if ( v3 == *(_QWORD *)result )
    {
LABEL_3:
      if ( result != v5 + 16 * v4 )
      {
        *(_QWORD *)(*(_QWORD *)(result + 8) + 8LL) = a2;
        *a2 = *(_QWORD *)(result + 8);
        *(_QWORD *)(result + 8) = a2;
      }
    }
    else
    {
      result = 1;
      while ( v7 != -1 )
      {
        v8 = result + 1;
        v6 = (v4 - 1) & (result + v6);
        result = v5 + 16LL * v6;
        v7 = *(_QWORD *)result;
        if ( v3 == *(_QWORD *)result )
          goto LABEL_3;
        result = v8;
      }
    }
  }
  return result;
}
