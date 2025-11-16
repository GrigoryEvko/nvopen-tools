// Function: sub_EA2ED0
// Address: 0xea2ed0
//
__int64 __fastcall sub_EA2ED0(__int64 a1, const void *a2, size_t a3, __int64 a4, __int64 a5)
{
  int v7; // eax
  unsigned int v8; // r8d
  _QWORD *v9; // r10
  __int64 result; // rax
  __int64 v11; // rax
  unsigned int v12; // r8d
  _QWORD *v13; // r10
  _QWORD *v14; // rcx
  __int64 *v15; // rdx
  _QWORD *v16; // [rsp+0h] [rbp-50h]
  _QWORD *v17; // [rsp+8h] [rbp-48h]
  unsigned int v18; // [rsp+14h] [rbp-3Ch]

  v7 = sub_C92610();
  v8 = sub_C92740(a1 + 344, a2, a3, v7);
  v9 = (_QWORD *)(*(_QWORD *)(a1 + 344) + 8LL * v8);
  result = *v9;
  if ( *v9 )
  {
    if ( result != -8 )
      goto LABEL_3;
    --*(_DWORD *)(a1 + 360);
  }
  v17 = v9;
  v18 = v8;
  v11 = sub_C7D670(a3 + 25, 8);
  v12 = v18;
  v13 = v17;
  v14 = (_QWORD *)v11;
  if ( a3 )
  {
    v16 = (_QWORD *)v11;
    memcpy((void *)(v11 + 24), a2, a3);
    v12 = v18;
    v13 = v17;
    v14 = v16;
  }
  *((_BYTE *)v14 + a3 + 24) = 0;
  *v14 = a3;
  v14[1] = 0;
  v14[2] = 0;
  *v13 = v14;
  ++*(_DWORD *)(a1 + 356);
  v15 = (__int64 *)(*(_QWORD *)(a1 + 344) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 344), v12));
  result = *v15;
  if ( *v15 )
    goto LABEL_9;
  do
  {
    do
    {
      result = v15[1];
      ++v15;
    }
    while ( !result );
LABEL_9:
    ;
  }
  while ( result == -8 );
LABEL_3:
  *(_QWORD *)(result + 16) = a5;
  *(_QWORD *)(result + 8) = a4;
  return result;
}
