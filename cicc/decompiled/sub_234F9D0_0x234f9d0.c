// Function: sub_234F9D0
// Address: 0x234f9d0
//
_QWORD *__fastcall sub_234F9D0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  int v6; // eax
  unsigned int v7; // ecx
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  unsigned int v11; // eax
  _QWORD *v12; // rdi
  int v13; // r14d
  _QWORD *v14; // rax
  unsigned int v15; // eax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  _QWORD *j; // rdx

  v3 = *(_QWORD *)(a2 + 8);
  v4 = (_QWORD *)sub_22077B0(0x10u);
  if ( v4 )
  {
    v4[1] = v3;
    *v4 = &unk_4A0B618;
LABEL_3:
    *a1 = v4;
    return a1;
  }
  if ( !v3 )
    goto LABEL_3;
  v6 = *(_DWORD *)(v3 + 80);
  ++*(_QWORD *)(v3 + 64);
  if ( v6 )
  {
    v7 = 4 * v6;
    v8 = *(unsigned int *)(v3 + 88);
    if ( (unsigned int)(4 * v6) < 0x40 )
      v7 = 64;
    if ( (unsigned int)v8 <= v7 )
    {
LABEL_9:
      v9 = *(_QWORD **)(v3 + 72);
      for ( i = &v9[3 * v8]; i != v9; *(v9 - 2) = -4096 )
      {
        *v9 = -4096;
        v9 += 3;
      }
      goto LABEL_11;
    }
    v11 = v6 - 1;
    if ( v11 )
    {
      _BitScanReverse(&v11, v11);
      v12 = *(_QWORD **)(v3 + 72);
      v13 = 1 << (33 - (v11 ^ 0x1F));
      if ( v13 < 64 )
        v13 = 64;
      if ( (_DWORD)v8 == v13 )
      {
        *(_QWORD *)(v3 + 80) = 0;
        v14 = &v12[3 * v8];
        do
        {
          if ( v12 )
          {
            *v12 = -4096;
            v12[1] = -4096;
          }
          v12 += 3;
        }
        while ( v14 != v12 );
        goto LABEL_12;
      }
    }
    else
    {
      v12 = *(_QWORD **)(v3 + 72);
      v13 = 64;
    }
    sub_C7D6A0((__int64)v12, 24 * v8, 8);
    v15 = sub_2309150(v13);
    *(_DWORD *)(v3 + 88) = v15;
    if ( v15 )
    {
      v16 = (_QWORD *)sub_C7D670(24LL * v15, 8);
      v17 = *(unsigned int *)(v3 + 88);
      *(_QWORD *)(v3 + 80) = 0;
      *(_QWORD *)(v3 + 72) = v16;
      for ( j = &v16[3 * v17]; j != v16; v16 += 3 )
      {
        if ( v16 )
        {
          *v16 = -4096;
          v16[1] = -4096;
        }
      }
      goto LABEL_12;
    }
  }
  else
  {
    if ( !*(_DWORD *)(v3 + 84) )
      goto LABEL_12;
    v8 = *(unsigned int *)(v3 + 88);
    if ( (unsigned int)v8 <= 0x40 )
      goto LABEL_9;
    sub_C7D6A0(*(_QWORD *)(v3 + 72), 24 * v8, 8);
    *(_DWORD *)(v3 + 88) = 0;
  }
  *(_QWORD *)(v3 + 72) = 0;
LABEL_11:
  *(_QWORD *)(v3 + 80) = 0;
LABEL_12:
  sub_BBC340(v3 + 32);
  *a1 = 0;
  return a1;
}
