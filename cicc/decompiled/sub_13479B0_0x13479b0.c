// Function: sub_13479B0
// Address: 0x13479b0
//
int __fastcall sub_13479B0(__int64 a1, __int64 a2, _QWORD **a3, bool *a4)
{
  _QWORD *v5; // r14
  _QWORD *i; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r8
  bool v13; // al
  __int64 v16; // [rsp+20h] [rbp-50h]
  unsigned __int64 v17; // [rsp+28h] [rbp-48h]
  _QWORD v18[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = *a3;
  do
  {
    if ( !v5 )
      break;
    v5[1] &= 0xFFFFFFFFFFFFF000LL;
    *v5 &= ~0x8000uLL;
    sub_1341E90(a1, *(_QWORD *)(a2 + 5616), (__int64)v5);
    v5 = (_QWORD *)v5[5];
  }
  while ( v5 != *a3 );
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 128)) )
  {
    sub_130AD90(a2 + 64);
    *(_BYTE *)(a2 + 168) = 1;
  }
  ++*(_QWORD *)(a2 + 120);
  if ( a1 != *(_QWORD *)(a2 + 112) )
  {
    ++*(_QWORD *)(a2 + 104);
    *(_QWORD *)(a2 + 112) = a1;
  }
  for ( i = *a3; *a3; i = *a3 )
  {
    v9 = (_QWORD *)i[5];
    *a3 = v9;
    if ( i == v9 )
    {
      *a3 = 0;
    }
    else
    {
      *(_QWORD *)(i[6] + 40LL) = v9[6];
      v10 = i[6];
      *(_QWORD *)(i[5] + 48LL) = v10;
      i[6] = *(_QWORD *)(v10 + 40);
      *(_QWORD *)(*(_QWORD *)(i[5] + 48LL) + 40LL) = i[5];
      *(_QWORD *)(i[6] + 40LL) = i;
    }
    v11 = i[3];
    v16 = i[1];
    v17 = i[2] & 0xFFFFFFFFFFFFF000LL;
    sub_1340D30(a1, (__int64 *)(a2 + 296), i);
    sub_134BCA0(a2 + 320, v11);
    sub_1349DA0(v11, v16, v17);
    if ( (*(_DWORD *)(v11 + 32) & 0xFFFF00) != 0 )
    {
      *(_WORD *)(v11 + 19) = 0;
    }
    else
    {
      v7 = *(_QWORD *)(v11 + 104);
      *(_BYTE *)(v11 + 19) = *(_QWORD *)(v11 + 176) != v7;
      if ( (unsigned __int64)(v7 << 12) >= *(_QWORD *)(a2 + 5632) && !*(_BYTE *)(v11 + 16) )
      {
        (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)(a2 + 56) + 296LL))(v18, 1);
        v8 = v18[0];
        *(_BYTE *)(v11 + 20) = 1;
        *(_QWORD *)(v11 + 24) = v8;
        v7 = *(_QWORD *)(v11 + 104);
      }
      if ( !v7 )
        *(_BYTE *)(v11 + 20) = 0;
    }
    sub_134BD00(a2 + 320, v11);
  }
  sub_1347550(a1, a2, 0);
  v12 = sub_134BFA0(a2 + 320);
  v13 = 1;
  if ( !v12 )
    v13 = sub_1347320(a2);
  *a4 = v13;
  *(_BYTE *)(a2 + 168) = 0;
  return pthread_mutex_unlock((pthread_mutex_t *)(a2 + 128));
}
