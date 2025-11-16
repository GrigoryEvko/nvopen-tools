// Function: sub_39BA9D0
// Address: 0x39ba9d0
//
__int64 __fastcall sub_39BA9D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // edx
  __int64 v4; // r8
  int v5; // esi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // rdi
  int v10; // edx
  int v11; // r9d

  v2 = sub_3981E80(*(_QWORD *)(a2 + 8));
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 600LL);
  if ( !v3 )
    goto LABEL_8;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 584LL);
  v5 = v3 - 1;
  v6 = (v3 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( v2 != *v7 )
  {
    v10 = 1;
    while ( v8 != -8 )
    {
      v11 = v10 + 1;
      v6 = v5 & (v10 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( v2 == *v7 )
        return *(unsigned int *)(v7[1] + 600);
      v10 = v11;
    }
LABEL_8:
    BUG();
  }
  return *(unsigned int *)(v7[1] + 600);
}
