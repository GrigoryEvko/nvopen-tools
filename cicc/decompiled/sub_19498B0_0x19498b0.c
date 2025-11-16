// Function: sub_19498B0
// Address: 0x19498b0
//
__int64 __fastcall sub_19498B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rcx
  unsigned int v5; // edi
  __int64 v6; // rdx
  __int64 v7; // r8
  int v8; // edx
  int v9; // r10d

  v2 = *(unsigned int *)(*(_QWORD *)a1 + 48LL);
  if ( !(_DWORD)v2 )
    return a2;
  v4 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = v4 + ((unsigned __int64)v5 << 6);
  v7 = *(_QWORD *)(v6 + 24);
  if ( v7 != a2 )
  {
    v8 = 1;
    while ( v7 != -8 )
    {
      v9 = v8 + 1;
      v5 = (v2 - 1) & (v8 + v5);
      v6 = v4 + ((unsigned __int64)v5 << 6);
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 == a2 )
        goto LABEL_4;
      v8 = v9;
    }
    return a2;
  }
LABEL_4:
  if ( v6 == v4 + (v2 << 6) )
    return a2;
  return *(_QWORD *)(v6 + 56);
}
