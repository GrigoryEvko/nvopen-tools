// Function: sub_19E1ED0
// Address: 0x19e1ed0
//
__int64 __fastcall sub_19E1ED0(__int64 a1, __int64 ***a2)
{
  int v2; // eax
  int v3; // ecx
  __int64 v4; // r8
  unsigned int v5; // edx
  __int64 ****v6; // rax
  __int64 ***v7; // r9
  __int64 ***v8; // rdx
  __int64 result; // rax
  int v10; // eax
  int v11; // r10d

  v2 = *(_DWORD *)(a1 + 1496);
  if ( !v2 )
    return (__int64)a2;
  v3 = v2 - 1;
  v4 = *(_QWORD *)(a1 + 1480);
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 ****)(v4 + 16LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v10 = 1;
    while ( v7 != (__int64 ***)-8LL )
    {
      v11 = v10 + 1;
      v5 = v3 & (v10 + v5);
      v6 = (__int64 ****)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        goto LABEL_3;
      v10 = v11;
    }
    return (__int64)a2;
  }
LABEL_3:
  v8 = v6[1];
  if ( !v8 )
    return (__int64)a2;
  if ( *(__int64 ****)(a1 + 1432) == v8 )
    return sub_1599EF0(*a2);
  result = (__int64)v8[4];
  if ( !result )
    return (__int64)v8[1];
  return result;
}
