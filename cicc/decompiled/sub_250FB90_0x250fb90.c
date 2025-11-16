// Function: sub_250FB90
// Address: 0x250fb90
//
__int64 __fastcall sub_250FB90(__int64 a1, __int64 a2)
{
  void *v2; // rdx
  __int64 (__fastcall *v3)(__int64); // rax
  char v4; // al
  __int64 v5; // rbx
  __int64 v6; // r13
  _WORD *v7; // rdx
  _DWORD *v8; // rdx

  v2 = *(void **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v2 <= 0xCu )
  {
    sub_CB6200(a1, "set-state(< {", 0xDu);
    v3 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL);
    if ( v3 == sub_2505E40 )
      goto LABEL_3;
  }
  else
  {
    qmemcpy(v2, "set-state(< {", 13);
    *(_QWORD *)(a1 + 32) += 13LL;
    v3 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 16LL);
    if ( v3 == sub_2505E40 )
    {
LABEL_3:
      v4 = *(_BYTE *)(a2 + 17);
      goto LABEL_4;
    }
  }
  v4 = v3(a2);
LABEL_4:
  if ( v4 )
  {
    v5 = *(_QWORD *)(a2 + 56);
    v6 = v5 + 16LL * *(unsigned int *)(a2 + 64);
    while ( v6 != v5 )
    {
      while ( 1 )
      {
        sub_C49420(v5, a1, 1);
        v7 = *(_WORD **)(a1 + 32);
        if ( *(_QWORD *)(a1 + 24) - (_QWORD)v7 <= 1u )
          break;
        v5 += 16;
        *v7 = 8236;
        *(_QWORD *)(a1 + 32) += 2LL;
        if ( v6 == v5 )
          goto LABEL_10;
      }
      v5 += 16;
      sub_CB6200(a1, (unsigned __int8 *)", ", 2u);
    }
LABEL_10:
    if ( *(_BYTE *)(a2 + 200) )
      sub_904010(a1, "undef ");
  }
  else
  {
    sub_904010(a1, "full-set");
  }
  v8 = *(_DWORD **)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) - (_QWORD)v8 <= 3u )
  {
    sub_CB6200(a1, "} >)", 4u);
  }
  else
  {
    *v8 = 691937405;
    *(_QWORD *)(a1 + 32) += 4LL;
  }
  return a1;
}
