// Function: sub_1F5BB60
// Address: 0x1f5bb60
//
unsigned __int64 __fastcall sub_1F5BB60(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 result; // rax
  __int64 v10; // rcx
  _DWORD *v11; // rdx
  int k; // ecx
  __int64 v13; // rcx
  _DWORD *v14; // rdx
  _DWORD *v15; // rax
  int j; // ecx
  __int64 v17; // rcx
  _DWORD *v18; // rdx
  _DWORD *v19; // rax
  int i; // ecx

  v6 = *(unsigned int *)(*(_QWORD *)(*((_QWORD *)a1 + 32) + 40LL) + 32LL);
  v7 = (unsigned int)a1[68];
  if ( v6 >= v7 )
  {
    if ( v6 <= v7 )
      goto LABEL_3;
    if ( v6 > (unsigned int)a1[69] )
    {
      sub_16CD150((__int64)(a1 + 66), a1 + 70, v6, 4, a5, a6);
      v7 = (unsigned int)a1[68];
    }
    v17 = *((_QWORD *)a1 + 33);
    v18 = (_DWORD *)(v17 + 4 * v6);
    v19 = (_DWORD *)(v17 + 4 * v7);
    for ( i = a1[70]; v18 != v19; ++v19 )
      *v19 = i;
  }
  a1[68] = v6;
LABEL_3:
  v8 = (unsigned int)a1[74];
  if ( v6 >= v8 )
  {
    if ( v6 <= v8 )
      goto LABEL_5;
    if ( v6 > (unsigned int)a1[75] )
    {
      sub_16CD150((__int64)(a1 + 72), a1 + 76, v6, 4, a5, a6);
      v8 = (unsigned int)a1[74];
    }
    v13 = *((_QWORD *)a1 + 36);
    v14 = (_DWORD *)(v13 + 4 * v6);
    v15 = (_DWORD *)(v13 + 4 * v8);
    for ( j = a1[76]; v14 != v15; ++v15 )
      *v15 = j;
  }
  a1[74] = v6;
LABEL_5:
  result = (unsigned int)a1[80];
  if ( v6 >= result )
  {
    if ( v6 <= result )
      return result;
    if ( v6 > (unsigned int)a1[81] )
    {
      sub_16CD150((__int64)(a1 + 78), a1 + 82, v6, 4, a5, a6);
      result = (unsigned int)a1[80];
    }
    v10 = *((_QWORD *)a1 + 39);
    v11 = (_DWORD *)(v10 + 4 * v6);
    result = v10 + 4 * result;
    for ( k = a1[82]; v11 != (_DWORD *)result; result += 4LL )
      *(_DWORD *)result = k;
  }
  a1[80] = v6;
  return result;
}
