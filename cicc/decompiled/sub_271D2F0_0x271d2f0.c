// Function: sub_271D2F0
// Address: 0x271d2f0
//
unsigned __int64 __fastcall sub_271D2F0(__int64 a1, char a2)
{
  __int64 v3; // r9
  __int64 v4; // rax
  unsigned __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rax
  __int64 v10; // rsi
  unsigned __int8 *v11; // r14
  unsigned __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // r13
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax

  sub_271D2E0(*(_QWORD *)a1, a2);
  v4 = **(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)v4 == 34 )
  {
    v20 = sub_AA5190(**(_QWORD **)(a1 + 16));
    v5 = v20;
    if ( !v20
      || (v6 = *(_QWORD *)(a1 + 16), v20 == *(_QWORD *)v6 + 48LL)
      && (v5 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      BUG();
    }
    if ( *(_BYTE *)(v5 - 24) == 39 )
    {
      *(_BYTE *)(*(_QWORD *)a1 + 120LL) = 1;
      v6 = *(_QWORD *)(a1 + 16);
    }
  }
  else
  {
    v5 = *(_QWORD *)(v4 + 32);
    v6 = *(_QWORD *)(a1 + 16);
  }
  if ( v5 == *(_QWORD *)v6 + 48LL )
  {
    v8 = *(_QWORD *)a1;
  }
  else
  {
    v7 = sub_AA5FF0(v5);
    v8 = *(_QWORD *)a1;
    v5 = v7;
    if ( !v7 )
      goto LABEL_6;
  }
  v5 -= 24LL;
LABEL_6:
  if ( !*(_BYTE *)(v8 + 100) )
  {
LABEL_16:
    sub_C8CC70(v8 + 72, v5, v6, v8, v8 + 72, v3);
    goto LABEL_11;
  }
  v9 = *(_QWORD **)(v8 + 80);
  v10 = *(unsigned int *)(v8 + 92);
  v6 = (__int64)&v9[v10];
  if ( v9 == (_QWORD *)v6 )
  {
LABEL_15:
    if ( (unsigned int)v10 >= *(_DWORD *)(v8 + 88) )
      goto LABEL_16;
    *(_DWORD *)(v8 + 92) = v10 + 1;
    *(_QWORD *)v6 = v5;
    ++*(_QWORD *)(v8 + 72);
  }
  else
  {
    while ( v5 != *v9 )
    {
      if ( (_QWORD *)v6 == ++v9 )
        goto LABEL_15;
    }
  }
LABEL_11:
  v11 = **(unsigned __int8 ***)(a1 + 8);
  result = (unsigned int)*v11 - 34;
  if ( (unsigned __int8)(*v11 - 34) <= 0x33u )
  {
    v13 = 0x8000000000041LL;
    if ( _bittest64(&v13, result) )
    {
      result = **(_QWORD **)(*((_QWORD *)v11 + 10) + 16LL);
      if ( *(_BYTE *)(result + 8) != 7 && (v11[7] & 0x80u) != 0 )
      {
        result = sub_BD2BC0(**(_QWORD **)(a1 + 8));
        v15 = result + v14;
        if ( (v11[7] & 0x80u) != 0 )
        {
          result = sub_BD2BC0((__int64)v11);
          v15 -= result;
        }
        v16 = v15 >> 4;
        if ( (_DWORD)v16 )
        {
          v17 = 0;
          v18 = 16LL * (unsigned int)v16;
          while ( 1 )
          {
            v19 = 0;
            if ( (v11[7] & 0x80u) != 0 )
              v19 = sub_BD2BC0((__int64)v11);
            result = *(_QWORD *)(v19 + v17);
            if ( *(_DWORD *)(result + 8) == 6 )
              break;
            v17 += 16;
            if ( v17 == v18 )
              return result;
          }
          result = *(_QWORD *)a1;
          *(_BYTE *)(*(_QWORD *)a1 + 120LL) = 1;
        }
      }
    }
  }
  return result;
}
