// Function: sub_15217C0
// Address: 0x15217c0
//
__int64 __fastcall sub_15217C0(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdi
  unsigned __int64 v4; // rax
  unsigned int v5; // edx
  __int64 v6; // r8
  __int64 *v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r8
  unsigned int v11[9]; // [rsp+Ch] [rbp-24h] BYREF

  v3 = *(_QWORD *)a1;
  v4 = (__int64)(*(_QWORD *)(v3 + 640) - *(_QWORD *)(v3 + 632)) >> 4;
  if ( a2 < v4 )
    return sub_15197A0(v3, a2);
  v5 = *(_DWORD *)(v3 + 8);
  if ( **(_BYTE **)(a1 + 8) )
  {
    if ( a2 >= v5
      || (v6 = *(_QWORD *)(*(_QWORD *)v3 + 8LL * a2)) == 0
      || (unsigned __int8)(*(_BYTE *)v6 - 4) <= 0x1Eu && (*(_BYTE *)(v6 + 1) == 2 || *(_DWORD *)(v6 + 12)) )
    {
      v7 = *(__int64 **)(a1 + 24);
      v11[0] = a2;
      v8 = v7[6];
      if ( v8 == v7[8] - 16 )
      {
        sub_1516A00(v7, (int *)v11);
        v9 = v7[6];
      }
      else
      {
        if ( v8 )
        {
          *(_DWORD *)v8 = 259;
          *(_QWORD *)(v8 + 8) = 0;
          *(_DWORD *)(v8 + 4) = a2;
          v8 = v7[6];
        }
        v9 = v8 + 16;
        v7[6] = v9;
      }
      if ( v7[7] == v9 )
        v9 = *(_QWORD *)(v7[9] - 8) + 512LL;
      return v9 - 16;
    }
    return v6;
  }
  if ( a2 < v5 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)v3 + 8LL * a2);
    if ( v6 )
      return v6;
  }
  if ( a2 < ((__int64)(*(_QWORD *)(v3 + 664) - *(_QWORD *)(v3 + 656)) >> 3) + v4 )
  {
    sub_1517EB0(v3, **(_DWORD **)(a1 + 16));
    sub_15201C0(*(_QWORD *)a1, a2, *(_QWORD *)(a1 + 24));
    v6 = 0;
    if ( a2 < *(_DWORD *)(*(_QWORD *)a1 + 8LL) )
      return *(_QWORD *)(**(_QWORD **)a1 + 8LL * a2);
    return v6;
  }
  return sub_1517EB0(v3, a2);
}
