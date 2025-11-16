// Function: sub_399CED0
// Address: 0x399ced0
//
__int64 __fastcall sub_399CED0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // r13
  _QWORD *v7; // rdi
  __int64 (*v8)(); // rax
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  unsigned int v13; // ecx
  _QWORD *v14; // rax
  _QWORD *j; // rdx
  bool v16; // zf
  __int64 v17; // rax
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  int v20; // eax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // r14d
  unsigned __int64 v24; // r13
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *i; // rdx
  _QWORD *v28; // rax
  _QWORD v29[5]; // [rsp+8h] [rbp-28h] BYREF

  *(_QWORD *)(a1 + 4008) = a2;
  result = sub_1626D20(*a2);
  v5 = *(_QWORD *)(result + 8 * (5LL - *(unsigned int *)(result + 8)));
  if ( !*(_DWORD *)(v5 + 36) )
    return result;
  v6 = sub_3999410(a1, v5);
  v7 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL);
  v8 = *(__int64 (**)())(*v7 + 88LL);
  if ( v8 == sub_168DB60 )
    goto LABEL_4;
  v16 = (unsigned __int8)v8() == 0;
  v17 = *(_QWORD *)(a1 + 8);
  if ( v16 )
  {
    v7 = *(_QWORD **)(v17 + 256);
LABEL_4:
    *(_DWORD *)(v7[1] + 1164LL) = *(_DWORD *)(v6 + 600);
    goto LABEL_5;
  }
  *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v17 + 256) + 8LL) + 1164LL) = 0;
LABEL_5:
  v9 = *(_DWORD *)(a1 + 6568);
  ++*(_QWORD *)(a1 + 6552);
  if ( v9 )
  {
    v13 = 4 * v9;
    v10 = *(unsigned int *)(a1 + 6576);
    if ( (unsigned int)(4 * v9) < 0x40 )
      v13 = 64;
    if ( v13 >= (unsigned int)v10 )
      goto LABEL_16;
    v18 = *(_QWORD **)(a1 + 6560);
    v19 = v9 - 1;
    if ( v19 )
    {
      _BitScanReverse(&v19, v19);
      v20 = 1 << (33 - (v19 ^ 0x1F));
      if ( v20 < 64 )
        v20 = 64;
      if ( (_DWORD)v10 == v20 )
      {
        *(_QWORD *)(a1 + 6568) = 0;
        v28 = &v18[v10];
        do
        {
          if ( v18 )
            *v18 = -8;
          ++v18;
        }
        while ( v28 != v18 );
        goto LABEL_9;
      }
      v21 = (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
          | (4 * v20 / 3u + 1)
          | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)
          | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
            | (4 * v20 / 3u + 1)
            | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4);
      v22 = (v21 >> 8) | v21;
      v23 = (v22 | (v22 >> 16)) + 1;
      v24 = 8 * ((v22 | (v22 >> 16)) + 1);
    }
    else
    {
      v24 = 1024;
      v23 = 128;
    }
    j___libc_free_0((unsigned __int64)v18);
    *(_DWORD *)(a1 + 6576) = v23;
    v25 = (_QWORD *)sub_22077B0(v24);
    v26 = *(unsigned int *)(a1 + 6576);
    *(_QWORD *)(a1 + 6568) = 0;
    *(_QWORD *)(a1 + 6560) = v25;
    for ( i = &v25[v26]; i != v25; ++v25 )
    {
      if ( v25 )
        *v25 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 6572) )
  {
    v10 = *(unsigned int *)(a1 + 6576);
    if ( (unsigned int)v10 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 6560));
      *(_QWORD *)(a1 + 6560) = 0;
      *(_QWORD *)(a1 + 6568) = 0;
      *(_DWORD *)(a1 + 6576) = 0;
      goto LABEL_9;
    }
LABEL_16:
    v14 = *(_QWORD **)(a1 + 6560);
    for ( j = &v14[v10]; j != v14; ++v14 )
      *v14 = -8;
    *(_QWORD *)(a1 + 6568) = 0;
  }
LABEL_9:
  sub_3989EF0(a1);
  result = (__int64)sub_399CD50(
                      v29,
                      a1,
                      a2,
                      *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL) + 8LL) + 1164LL));
  v11 = *(_QWORD *)(a1 + 48);
  if ( v11 )
    result = sub_161E7C0(a1 + 48, v11);
  v12 = (unsigned __int8 *)v29[0];
  *(_QWORD *)(a1 + 48) = v29[0];
  if ( v12 )
    return sub_1623210((__int64)v29, v12, a1 + 48);
  return result;
}
