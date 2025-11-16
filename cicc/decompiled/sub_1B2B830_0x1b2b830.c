// Function: sub_1B2B830
// Address: 0x1b2b830
//
__int64 __fastcall sub_1B2B830(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // rax
  int v4; // ecx
  __int64 v5; // r8
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r9
  int v10; // eax
  int v11; // r10d

  v2 = *(_DWORD *)(a1 + 3224);
  v3 = 0;
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = *(_QWORD *)(a1 + 3208);
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      v3 = 96LL * *((unsigned int *)v7 + 2);
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v11 = v10 + 1;
        v6 = v4 & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v11;
      }
      v3 = 0;
    }
  }
  return *(_QWORD *)(a1 + 112) + v3;
}
