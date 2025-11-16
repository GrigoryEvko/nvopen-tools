// Function: sub_2E0EAB0
// Address: 0x2e0eab0
//
_QWORD *__fastcall sub_2E0EAB0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // r12
  unsigned __int64 v5; // r15
  __int64 v6; // r14
  signed __int64 v7; // rcx
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  unsigned int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // r12
  unsigned int v13; // eax
  int v15; // eax
  unsigned __int64 v17; // rdx
  unsigned int v18; // r15d
  unsigned __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rcx
  unsigned int v22; // eax
  unsigned __int64 v23; // [rsp+8h] [rbp-68h]
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  __int64 *v25; // [rsp+18h] [rbp-58h] BYREF
  __int64 v26[10]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *(_QWORD **)(a1 + 96);
  if ( v4 )
  {
    v26[0] = a1;
    if ( !v4[5] )
      return 0;
    v5 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    v6 = (a3 >> 1) & 3;
    v7 = ((a3 >> 1) & 3) != 0 ? v5 | (2LL * ((int)v6 - 1)) : *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL | 6;
    v8 = (_QWORD *)v4[2];
    if ( v8 )
    {
      v9 = (__int64)(v4 + 1);
      v10 = *(_DWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v7 >> 1) & 3;
      do
      {
        while ( 1 )
        {
          v11 = *(_DWORD *)((v8[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v8[4] >> 1) & 3;
          if ( v10 < v11
            || v10 <= v11
            && ((unsigned int)v6 | *(_DWORD *)(v5 + 24)) < (*(_DWORD *)((v8[5] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                          | (unsigned int)((__int64)v8[5] >> 1) & 3) )
          {
            break;
          }
          v8 = (_QWORD *)v8[3];
          if ( !v8 )
            goto LABEL_12;
        }
        v9 = (__int64)v8;
        v8 = (_QWORD *)v8[2];
      }
      while ( v8 );
LABEL_12:
      if ( v4 + 1 != (_QWORD *)v9
        && v10 >= (*(_DWORD *)((*(_QWORD *)(v9 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(*(__int64 *)(v9 + 32) >> 1) & 3) )
      {
        v9 = sub_220EF30(v9);
      }
    }
    else
    {
      v9 = (__int64)(v4 + 1);
    }
    if ( v4[3] == v9 )
      return 0;
    v12 = sub_220EFE0(v9);
    v13 = *(_DWORD *)((*(_QWORD *)(v12 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v12 + 40) >> 1) & 3;
    if ( v13 <= (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
    {
      return 0;
    }
    else
    {
      if ( v13 < (*(_DWORD *)(v5 + 24) | (unsigned int)v6) )
        sub_2E0E620((__int64)v26, v12, a3);
      return *(_QWORD **)(v12 + 48);
    }
  }
  else
  {
    v15 = *(_DWORD *)(a1 + 8);
    v25 = (__int64 *)a1;
    if ( v15 )
    {
      v17 = a3 & 0xFFFFFFFFFFFFFFF8LL;
      v18 = (a3 >> 1) & 3;
      v19 = v18 ? v17 | (2LL * (int)(v18 - 1)) : *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL | 6;
      v23 = v17;
      v26[0] = v19;
      v26[1] = a3;
      v26[2] = 0;
      v20 = sub_2E09C80(a1, v26);
      v21 = v20;
      if ( v20 != *(_QWORD **)a1 )
      {
        v22 = *(_DWORD *)((*(v20 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)*(v20 - 2) >> 1) & 3;
        if ( v22 > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) )
        {
          if ( v22 < (*(_DWORD *)(v23 + 24) | v18) )
          {
            v24 = v21;
            sub_2E097D0(&v25, v21 - 3, a3);
            v21 = v24;
          }
          return (_QWORD *)*(v21 - 1);
        }
      }
    }
  }
  return v4;
}
