// Function: sub_1832310
// Address: 0x1832310
//
__int64 __fastcall sub_1832310(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v6; // rdi
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx

  v6 = *(_QWORD **)a1;
  v7 = *(_QWORD *)(*(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL);
  v8 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v9 = (a3 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v10 = (unsigned __int64 *)(v8 - 72);
  if ( (a3 & 4) != 0 )
    v10 = (unsigned __int64 *)v9;
  v11 = *v10;
  if ( *(_BYTE *)(*v10 + 16) )
    v11 = 0;
  v12 = sub_1399010(v6, v11);
  v13 = *(_QWORD *)a1 + 16LL;
  v14 = *(_QWORD **)(*(_QWORD *)a1 + 24LL);
  if ( v14 )
  {
    v15 = *(_QWORD *)a1 + 16LL;
    do
    {
      while ( 1 )
      {
        v16 = v14[2];
        v17 = v14[3];
        if ( v14[4] >= v7 )
          break;
        v14 = (_QWORD *)v14[3];
        if ( !v17 )
          goto LABEL_10;
      }
      v15 = (__int64)v14;
      v14 = (_QWORD *)v14[2];
    }
    while ( v16 );
LABEL_10:
    if ( v13 != v15 && *(_QWORD *)(v15 + 32) <= v7 )
      v13 = v15;
  }
  return sub_13986C0(*(_QWORD *)(v13 + 40), a2, a3, v12);
}
