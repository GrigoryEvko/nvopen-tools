// Function: sub_33D2050
// Address: 0x33d2050
//
__int64 __fastcall sub_33D2050(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v9; // r13d
  int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rdi
  int v14; // ecx
  unsigned int v15; // r14d
  bool v16; // al
  __int64 result; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  int v20; // esi
  unsigned int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rdi
  _DWORD *v24; // rax
  __int64 v25; // r10
  int v26; // eax
  unsigned int v27; // eax
  __int64 v30; // [rsp+8h] [rbp-38h]

  v9 = *(_DWORD *)(a1 + 64);
  if ( !a3 )
    goto LABEL_7;
  *(_DWORD *)(a3 + 8) = 0;
  LOBYTE(v10) = v9;
  v11 = (unsigned int)(v9 + 63) >> 6;
  *(_DWORD *)(a3 + 64) = v9;
  if ( v11 )
  {
    v12 = v11;
    v13 = 0;
    if ( *(_DWORD *)(a3 + 12) < v11 )
    {
      v30 = v11;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v11, 8u, a5, a6);
      v12 = v30;
      v13 = 8LL * *(unsigned int *)(a3 + 8);
    }
    memset((void *)(*(_QWORD *)a3 + v13), 0, 8 * v12);
    *(_DWORD *)(a3 + 8) += (unsigned int)(v9 + 63) >> 6;
    v10 = *(_DWORD *)(a3 + 64);
  }
  v14 = v10 & 0x3F;
  if ( v14 )
  {
    *(_QWORD *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8) - 8) &= ~(-1LL << v14);
    v15 = *(_DWORD *)(a2 + 8);
    if ( v15 > 0x40 )
      goto LABEL_8;
  }
  else
  {
LABEL_7:
    v15 = *(_DWORD *)(a2 + 8);
    if ( v15 > 0x40 )
    {
LABEL_8:
      v16 = v15 == (unsigned int)sub_C444A0(a2);
      goto LABEL_9;
    }
  }
  v16 = *(_QWORD *)a2 == 0;
LABEL_9:
  if ( v16 )
    return 0;
  if ( !v9 )
    goto LABEL_30;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  while ( 1 )
  {
    v22 = *(_QWORD *)a2;
    v23 = 1LL << v21;
    if ( v15 > 0x40 )
      v22 = *(_QWORD *)(v22 + 8LL * (v21 >> 6));
    if ( (v22 & v23) != 0 )
    {
      v24 = (_DWORD *)(v18 + *(_QWORD *)(a1 + 40));
      v25 = *(_QWORD *)v24;
      if ( *(_DWORD *)(*(_QWORD *)v24 + 24LL) == 51 )
      {
        if ( a3 )
          *(_QWORD *)(*(_QWORD *)a3 + 8LL * (v21 >> 6)) |= v23;
      }
      else
      {
        v26 = v24[2];
        if ( v19 )
        {
          if ( v25 != v19 || v26 != v20 )
            return 0;
        }
        else
        {
          v20 = v26;
          v19 = v25;
        }
      }
    }
    ++v21;
    v18 += 40;
    if ( v21 == v9 )
      break;
    v15 = *(_DWORD *)(a2 + 8);
  }
  result = v19;
  if ( !v19 )
  {
    v15 = *(_DWORD *)(a2 + 8);
LABEL_30:
    if ( v15 <= 0x40 )
    {
      _RAX = *(_QWORD *)a2;
      __asm { tzcnt   rdx, rax }
      v27 = 64;
      if ( *(_QWORD *)a2 )
        v27 = _RDX;
      if ( v15 <= v27 )
        v27 = v15;
    }
    else
    {
      v27 = sub_C44590(a2);
    }
    return *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL * v27);
  }
  return result;
}
