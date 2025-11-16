// Function: sub_20C2240
// Address: 0x20c2240
//
__int64 __fastcall sub_20C2240(__int64 a1, unsigned int a2, __int64 a3)
{
  size_t v4; // r12
  char *v5; // rax
  char *v6; // r14
  char *v7; // rax
  char *v8; // r14
  char *v9; // rax
  char *v10; // r14
  char *v11; // rax
  char *v12; // r14
  __int64 v13; // rdx
  int v14; // eax
  int v15; // ecx
  __int64 result; // rax
  __int64 v17; // rdx

  *(_DWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( a2 )
  {
    v4 = 4LL * a2;
    v5 = (char *)sub_22077B0(v4);
    v6 = &v5[v4];
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 24) = &v5[v4];
    memset(v5, 0, v4);
    *(_QWORD *)(a1 + 16) = v6;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 48) = 0;
    v7 = (char *)sub_22077B0(v4);
    v8 = &v7[v4];
    *(_QWORD *)(a1 + 32) = v7;
    *(_QWORD *)(a1 + 48) = &v7[v4];
    memset(v7, 0, v4);
    *(_QWORD *)(a1 + 40) = v8;
    *(_DWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 80) = a1 + 64;
    *(_QWORD *)(a1 + 88) = a1 + 64;
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 112) = 0;
    *(_QWORD *)(a1 + 120) = 0;
    v9 = (char *)sub_22077B0(v4);
    v10 = &v9[v4];
    *(_QWORD *)(a1 + 104) = v9;
    *(_QWORD *)(a1 + 120) = &v9[v4];
    memset(v9, 0, v4);
    *(_QWORD *)(a1 + 112) = v10;
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 136) = 0;
    *(_QWORD *)(a1 + 144) = 0;
    v11 = (char *)sub_22077B0(v4);
    v12 = &v11[v4];
    *(_QWORD *)(a1 + 128) = v11;
    *(_QWORD *)(a1 + 144) = &v11[v4];
    memset(v11, 0, v4);
  }
  else
  {
    *(_QWORD *)(a1 + 32) = 0;
    v12 = 0;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_DWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 80) = a1 + 64;
    *(_QWORD *)(a1 + 88) = a1 + 64;
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 112) = 0;
    *(_QWORD *)(a1 + 120) = 0;
    *(_QWORD *)(a1 + 128) = 0;
    *(_QWORD *)(a1 + 144) = 0;
  }
  v13 = *(_QWORD *)(a3 + 32);
  *(_QWORD *)(a1 + 136) = v12;
  if ( v13 == a3 + 24 )
  {
    v15 = 0;
  }
  else
  {
    v14 = 0;
    do
    {
      v13 = *(_QWORD *)(v13 + 8);
      ++v14;
    }
    while ( v13 != a3 + 24 );
    v15 = v14;
  }
  for ( result = 0; *(_DWORD *)a1 > (unsigned int)result; *(_DWORD *)(*(_QWORD *)(a1 + 128) + 4 * v17) = v15 )
  {
    v17 = (unsigned int)result;
    *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * (unsigned int)result) = result;
    result = (unsigned int)(result + 1);
    *(_DWORD *)(*(_QWORD *)(a1 + 104) + 4 * v17) = -1;
  }
  return result;
}
