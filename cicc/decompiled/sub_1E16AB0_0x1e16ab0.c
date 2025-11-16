// Function: sub_1E16AB0
// Address: 0x1e16ab0
//
__int64 __fastcall sub_1E16AB0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  unsigned int v11; // r13d
  int v12; // edx
  _BYTE *v13; // rcx
  int v14; // r8d
  __int64 v15; // rax
  unsigned int v16; // r12d
  __int64 v17; // rax
  int v18; // ecx
  unsigned int v19; // eax
  int v20; // r10d
  int v21; // [rsp+Ch] [rbp-74h]
  unsigned int v22; // [rsp+18h] [rbp-68h]
  _BYTE *v23; // [rsp+20h] [rbp-60h] BYREF
  __int64 v24; // [rsp+28h] [rbp-58h]
  _BYTE v25[80]; // [rsp+30h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a1 + 32);
  v7 = v6 + 40LL * a2;
  if ( (*(_WORD *)(v7 + 2) & 0xFF0) != 0xFF0 )
    return (unsigned int)(unsigned __int8)(*(_WORD *)(v7 + 2) >> 4) - 1;
  if ( **(_WORD **)(a1 + 16) != 1 )
  {
    result = 254;
    if ( (*(_BYTE *)(v7 + 3) & 0x10) != 0 )
    {
      while ( 1 )
      {
        v9 = v6 + 40LL * (unsigned int)result;
        if ( !*(_BYTE *)v9 && (*(_BYTE *)(v9 + 3) & 0x10) == 0 && (unsigned __int8)(*(_WORD *)(v9 + 2) >> 4) == a2 + 1 )
          break;
        result = (unsigned int)(result + 1);
      }
    }
    return result;
  }
  v10 = v6 + 80;
  v11 = 2;
  v12 = 0;
  v23 = v25;
  v13 = v25;
  v14 = -1;
  v24 = 0x800000000LL;
  v15 = 0;
  while ( 1 )
  {
    *(_DWORD *)&v13[4 * v15] = v11;
    v16 = v24 + 1;
    v17 = *(_QWORD *)(v10 + 24);
    LODWORD(v24) = v24 + 1;
    v18 = ((unsigned __int16)v17 >> 3) + 1;
    if ( v11 >= a2 )
      break;
    LODWORD(a6) = v11 + v18;
    if ( v11 + v18 <= a2 )
      break;
    if ( (int)v17 < 0 )
    {
      a6 = v23;
      v20 = *(_DWORD *)&v23[4 * (WORD1(v17) & 0x7FFF)];
LABEL_25:
      result = a2 + v20 - v11;
      goto LABEL_26;
    }
    v11 += v18;
    v14 = v12;
LABEL_16:
    v10 = *(_QWORD *)(a1 + 32) + 40LL * v11;
    v15 = v16;
    if ( HIDWORD(v24) <= v16 )
    {
      v21 = v14;
      sub_16CD150((__int64)&v23, v25, 0, 4, v14, (int)a6);
      v15 = (unsigned int)v24;
      v14 = v21;
    }
    v13 = v23;
    v12 = v16;
  }
  if ( (int)v17 >= 0 )
    goto LABEL_23;
  a6 = v23;
  v19 = WORD1(v17) & 0x7FFF;
  v20 = *(_DWORD *)&v23[4 * v19];
  if ( v14 == v12 )
    goto LABEL_25;
  if ( v14 != v19 )
  {
LABEL_23:
    v11 += v18;
    goto LABEL_16;
  }
  result = v11 + a2 - v20;
LABEL_26:
  if ( a6 != v25 )
  {
    v22 = result;
    _libc_free((unsigned __int64)a6);
    return v22;
  }
  return result;
}
