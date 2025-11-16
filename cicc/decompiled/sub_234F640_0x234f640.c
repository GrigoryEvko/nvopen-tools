// Function: sub_234F640
// Address: 0x234f640
//
void *__fastcall sub_234F640(_QWORD *a1)
{
  void *result; // rax
  __int64 v2; // rbx
  int v3; // eax
  unsigned int v4; // ecx
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  unsigned int v8; // eax
  _QWORD *v9; // rdi
  __int64 v10; // r12
  _QWORD *v11; // rax
  unsigned int v12; // eax
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *j; // rdx

  result = &unk_4A0B618;
  v2 = a1[1];
  *a1 = &unk_4A0B618;
  if ( v2 )
  {
    v3 = *(_DWORD *)(v2 + 80);
    ++*(_QWORD *)(v2 + 64);
    if ( v3 )
    {
      v4 = 4 * v3;
      v5 = *(unsigned int *)(v2 + 88);
      if ( (unsigned int)(4 * v3) < 0x40 )
        v4 = 64;
      if ( v4 >= (unsigned int)v5 )
      {
LABEL_6:
        v6 = *(_QWORD **)(v2 + 72);
        for ( i = &v6[3 * v5]; i != v6; *(v6 - 2) = -4096 )
        {
          *v6 = -4096;
          v6 += 3;
        }
        goto LABEL_8;
      }
      v8 = v3 - 1;
      if ( v8 )
      {
        _BitScanReverse(&v8, v8);
        v9 = *(_QWORD **)(v2 + 72);
        v10 = (unsigned int)(1 << (33 - (v8 ^ 0x1F)));
        if ( (int)v10 < 64 )
          v10 = 64;
        if ( (_DWORD)v10 == (_DWORD)v5 )
        {
          *(_QWORD *)(v2 + 80) = 0;
          v11 = &v9[3 * v10];
          do
          {
            if ( v9 )
            {
              *v9 = -4096;
              v9[1] = -4096;
            }
            v9 += 3;
          }
          while ( v11 != v9 );
          return (void *)sub_BBC340(v2 + 32);
        }
      }
      else
      {
        v9 = *(_QWORD **)(v2 + 72);
        LODWORD(v10) = 64;
      }
      sub_C7D6A0((__int64)v9, 24 * v5, 8);
      v12 = sub_2309150(v10);
      *(_DWORD *)(v2 + 88) = v12;
      if ( v12 )
      {
        v13 = (_QWORD *)sub_C7D670(24LL * v12, 8);
        v14 = *(unsigned int *)(v2 + 88);
        *(_QWORD *)(v2 + 80) = 0;
        *(_QWORD *)(v2 + 72) = v13;
        for ( j = &v13[3 * v14]; j != v13; v13 += 3 )
        {
          if ( v13 )
          {
            *v13 = -4096;
            v13[1] = -4096;
          }
        }
        return (void *)sub_BBC340(v2 + 32);
      }
    }
    else
    {
      if ( !*(_DWORD *)(v2 + 84) )
        return (void *)sub_BBC340(v2 + 32);
      v5 = *(unsigned int *)(v2 + 88);
      if ( (unsigned int)v5 <= 0x40 )
        goto LABEL_6;
      sub_C7D6A0(*(_QWORD *)(v2 + 72), 24 * v5, 8);
      *(_DWORD *)(v2 + 88) = 0;
    }
    *(_QWORD *)(v2 + 72) = 0;
LABEL_8:
    *(_QWORD *)(v2 + 80) = 0;
    return (void *)sub_BBC340(v2 + 32);
  }
  return result;
}
