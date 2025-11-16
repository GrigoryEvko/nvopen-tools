// Function: sub_30D16B0
// Address: 0x30d16b0
//
__int64 __fastcall sub_30D16B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  unsigned int v4; // edx
  __int64 *v5; // rax
  __int64 v6; // r10
  __int64 result; // rax
  int v8; // eax
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 816);
  v3 = *(_QWORD *)(a1 + 800);
  if ( !(_DWORD)v2 )
  {
LABEL_6:
    v5 = (__int64 *)(v3 + 16 * v2);
    goto LABEL_3;
  }
  v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = (__int64 *)(v3 + 16LL * v4);
  v6 = *v5;
  if ( a2 != *v5 )
  {
    v8 = 1;
    while ( v6 != -4096 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        goto LABEL_3;
      v8 = v9;
    }
    goto LABEL_6;
  }
LABEL_3:
  *((_DWORD *)v5 + 2) += qword_5030168;
  result = (unsigned int)qword_5030168;
  *(_DWORD *)(a1 + 780) += qword_5030168;
  return result;
}
