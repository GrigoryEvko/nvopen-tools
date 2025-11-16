// Function: sub_20421A0
// Address: 0x20421a0
//
__int64 __fastcall sub_20421A0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  int v4; // edx
  unsigned int *v5; // rdx
  int v7; // eax
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  int v13; // eax

  if ( a2 )
  {
    v2 = *a2;
    if ( *a2 )
    {
      v4 = *(_DWORD *)(v2 + 56);
      if ( v4 )
      {
        v5 = (unsigned int *)(*(_QWORD *)(v2 + 32) + 40LL * (unsigned int)(v4 - 1));
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v5 + 40LL) + 16LL * v5[2]) == 111 )
          return 1;
      }
      v7 = *(__int16 *)(v2 + 24);
      if ( (v7 & 0x8000u) == 0 )
      {
LABEL_7:
        v8 = a1[21];
        v9 = (a1[22] - v8) >> 3;
        if ( !(_DWORD)v9 )
          return 1;
        v10 = v8 + 8LL * (unsigned int)(v9 - 1) + 8;
        while ( 1 )
        {
          v11 = *(_QWORD **)(*(_QWORD *)v8 + 112LL);
          v12 = &v11[2 * *(unsigned int *)(*(_QWORD *)v8 + 120LL)];
          if ( v11 != v12 )
            break;
LABEL_13:
          v8 += 8;
          if ( v10 == v8 )
            return 1;
        }
        while ( (*v11 & 6) != 0 || a2 != (__int64 *)(*v11 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          v11 += 2;
          if ( v12 == v11 )
            goto LABEL_13;
        }
        return 0;
      }
      v13 = ~v7;
      if ( v13 > 10 )
      {
        if ( v13 == 14 )
          goto LABEL_7;
      }
      else if ( v13 > 6 )
      {
        goto LABEL_7;
      }
      if ( (unsigned __int8)sub_20E8AB0(a1[20], *(_QWORD *)(a1[18] + 8LL) + ((__int64)v13 << 6)) )
        goto LABEL_7;
    }
    return 0;
  }
  return 0;
}
