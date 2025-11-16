// Function: sub_2DF47B0
// Address: 0x2df47b0
//
__int64 __fastcall sub_2DF47B0(__int64 a1, int a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // ecx
  unsigned int v5; // edx
  int *v6; // rax
  int v7; // edi
  __int64 v8; // rcx
  __int64 result; // rax
  __int64 v10; // rdx
  int v11; // eax
  int v12; // r9d

  v2 = *(_DWORD *)(a1 + 1136);
  v3 = *(_QWORD *)(a1 + 1120);
  if ( !v2 )
    return 0;
  v4 = v2 - 1;
  v5 = (v2 - 1) & (37 * a2);
  v6 = (int *)(v3 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v11 = 1;
    while ( v7 != -1 )
    {
      v12 = v11 + 1;
      v5 = v4 & (v11 + v5);
      v6 = (int *)(v3 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_3;
      v11 = v12;
    }
    return 0;
  }
LABEL_3:
  v8 = *((_QWORD *)v6 + 1);
  if ( !v8 )
    return 0;
  result = *(_QWORD *)(v8 + 40);
  do
  {
    v10 = result;
    result = *(_QWORD *)(result + 40);
  }
  while ( v10 != result );
  *(_QWORD *)(v8 + 40) = result;
  return result;
}
