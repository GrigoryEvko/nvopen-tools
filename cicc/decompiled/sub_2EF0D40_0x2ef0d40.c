// Function: sub_2EF0D40
// Address: 0x2ef0d40
//
__int64 __fastcall sub_2EF0D40(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        char a7,
        __int64 a8,
        __int64 a9)
{
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // r8
  void *v18; // rdx
  _BYTE *v19; // rax
  unsigned __int8 v20; // dl
  __int64 result; // rax
  unsigned __int64 v22; // rbx
  __int64 *v23; // rdx
  __int64 v24; // rsi
  unsigned int v25; // edi
  unsigned int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-58h]
  __int128 v31; // [rsp+20h] [rbp-40h]

  *((_QWORD *)&v31 + 1) = a8;
  *(_QWORD *)&v31 = a9;
  v13 = (__int64 *)sub_2E09D00((__int64 *)a5, a4);
  if ( v13 == (__int64 *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8))
    || (v14 = (a4 >> 1) & 3,
        (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v13 >> 1) & 3) > ((unsigned int)v14
                                                                                           | *(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)))
    || (v15 = v13[2]) == 0 )
  {
    sub_2EF0A60(a1, "No live segment at def", a2, a3, 0);
    sub_2EEF4F0(*(_QWORD *)(a1 + 16), a5);
    sub_2EEFA20(a1, a6);
    if ( v31 != 0 )
      sub_2EEF800(*(_QWORD *)(a1 + 16), a8, a9);
    sub_2EEF640(*(_QWORD *)(a1 + 16), a4);
  }
  else
  {
    v16 = *(_QWORD *)(v15 + 8);
    if ( a7 || (*(_DWORD *)a2 & 0xFFF00) == 0 )
    {
      if ( v16 == a4 )
        goto LABEL_17;
    }
    else if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) == (v16 & 0xFFFFFFFFFFFFFFF8LL)
           && (v16 == a4 || ((v16 >> 1) & 3) == 1 && v14 == 2) )
    {
      goto LABEL_17;
    }
    sub_2EF0A60(a1, "Inconsistent valno->def", a2, a3, 0);
    v17 = *(_QWORD *)(a1 + 16);
    v18 = *(void **)(v17 + 32);
    if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 0xEu )
    {
      v17 = sub_CB6200(*(_QWORD *)(a1 + 16), "- liverange:   ", 0xFu);
    }
    else
    {
      qmemcpy(v18, "- liverange:   ", 15);
      *(_QWORD *)(v17 + 32) += 15LL;
    }
    v28 = v17;
    sub_2E0B3F0(a5, v17);
    v19 = *(_BYTE **)(v28 + 32);
    if ( (unsigned __int64)v19 >= *(_QWORD *)(v28 + 24) )
    {
      sub_CB5D20(v28, 10);
    }
    else
    {
      *(_QWORD *)(v28 + 32) = v19 + 1;
      *v19 = 10;
    }
    sub_2EEFA20(a1, a6);
    if ( v31 != 0 )
      sub_2EEF800(*(_QWORD *)(a1 + 16), a8, a9);
    sub_2EEF900(*(_QWORD *)(a1 + 16), (unsigned int *)v15);
    sub_2EEF640(*(_QWORD *)(a1 + 16), a4);
  }
LABEL_17:
  v20 = *(_BYTE *)(a2 + 3);
  result = (v20 & 0x10) != 0;
  if ( ((unsigned __int8)result & (v20 >> 6)) == 0 )
    return result;
  v22 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v23 = (__int64 *)sub_2E09D00((__int64 *)a5, v22);
  result = *(_QWORD *)a5;
  v24 = *(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8);
  if ( v23 != (__int64 *)v24 )
  {
    v25 = *(_DWORD *)(v22 + 24);
    v26 = *(_DWORD *)((*v23 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( (unsigned __int64)(v26 | (*v23 >> 1) & 3) <= v25 )
    {
      v27 = v23[1];
      if ( v22 == (v27 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        if ( (__int64 *)v24 == v23 + 3 )
          goto LABEL_26;
        v26 = *(_DWORD *)((v23[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v23 += 3;
      }
    }
    else
    {
      v27 = 0;
    }
    if ( v25 >= v26 )
      v27 = v23[1];
LABEL_26:
    result = v27 ^ 6;
    if ( (result & 6) == 0 )
      return result;
  }
  if ( a7 || (*(_DWORD *)a2 & 0xFFF00) == 0 )
  {
    sub_2EF0A60(a1, "Live range continues after dead def flag", a2, a3, 0);
    sub_2EEF4F0(*(_QWORD *)(a1 + 16), a5);
    sub_2EEFA20(a1, a6);
    result = a8 | a9;
    if ( v31 != 0 )
      return sub_2EEF800(*(_QWORD *)(a1 + 16), a8, a9);
  }
  return result;
}
