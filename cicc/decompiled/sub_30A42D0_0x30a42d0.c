// Function: sub_30A42D0
// Address: 0x30a42d0
//
__int64 __fastcall sub_30A42D0(__int64 a1, __int64 a2, __m128i si128)
{
  char v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  bool v8; // al
  bool v9; // cl
  __int64 v10; // rdi
  __m128 *v11; // rdx
  char *v12; // rax
  __int64 v13; // rdx
  bool v14; // al
  bool v15; // r11
  __int64 v16; // rdi
  _BYTE *v17; // rax
  char v19; // [rsp+Eh] [rbp-32h]
  bool v20; // [rsp+Fh] [rbp-31h]
  bool v21; // [rsp+Fh] [rbp-31h]
  bool v22; // [rsp+Fh] [rbp-31h]
  bool v23; // [rsp+Fh] [rbp-31h]

  v4 = sub_BC5DE0();
  v19 = v4 & sub_BC63A0("*", 1);
  if ( v19 )
  {
LABEL_26:
    sub_CB6200(*(_QWORD *)(a1 + 208), *(unsigned __int8 **)(a1 + 176), *(_QWORD *)(a1 + 184));
LABEL_20:
    v16 = *(_QWORD *)(a1 + 208);
    v17 = *(_BYTE **)(v16 + 32);
    if ( *(_BYTE **)(v16 + 24) == v17 )
    {
      sub_CB6200(v16, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v17 = 10;
      ++*(_QWORD *)(v16 + 32);
    }
    sub_A69980(**(__int64 (__fastcall ****)())a2, *(_QWORD *)(a1 + 208), 0, 0, 0, si128);
    return 0;
  }
  v20 = 0;
  v5 = *(_QWORD *)(a2 + 24);
  v6 = *(_QWORD *)(a2 + 16);
  if ( v5 == v6 )
    return 0;
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(*(_QWORD *)v6 + 8LL);
      if ( !v7 )
      {
        v8 = sub_BC63A0("*", 1);
        v9 = v8;
        if ( v8 )
        {
          if ( !v20 )
          {
            v21 = v8;
            sub_CB6200(*(_QWORD *)(a1 + 208), *(unsigned __int8 **)(a1 + 176), *(_QWORD *)(a1 + 184));
            v9 = v21;
          }
          v10 = *(_QWORD *)(a1 + 208);
          v11 = *(__m128 **)(v10 + 32);
          if ( *(_QWORD *)(v10 + 24) - (_QWORD)v11 <= 0x19u )
          {
            v22 = v9;
            sub_CB6200(v10, "\nPrinting <null> Function\n", 0x1Au);
            v9 = v22;
          }
          else
          {
            si128 = _mm_load_si128((const __m128i *)&xmmword_44CB540);
            qmemcpy(&v11[1], " Function\n", 10);
            *v11 = (__m128)si128;
            *(_QWORD *)(v10 + 32) += 26LL;
          }
          v20 = v9;
        }
        goto LABEL_5;
      }
      if ( !sub_B2FC80(*(_QWORD *)(*(_QWORD *)v6 + 8LL)) )
      {
        v12 = (char *)sub_BD5D20(v7);
        v14 = sub_BC63A0(v12, v13);
        v15 = v14;
        if ( v14 )
        {
          v19 = v4;
          if ( !v4 )
            break;
        }
      }
LABEL_5:
      v6 += 8;
      if ( v5 == v6 )
        goto LABEL_17;
    }
    if ( !v20 )
    {
      v23 = v14;
      sub_CB6200(*(_QWORD *)(a1 + 208), *(unsigned __int8 **)(a1 + 176), *(_QWORD *)(a1 + 184));
      v15 = v23;
    }
    v20 = v15;
    v6 += 8;
    sub_A68C30(v7, *(_QWORD *)(a1 + 208), 0, 0, 0);
    v19 = v20;
  }
  while ( v5 != v6 );
LABEL_17:
  if ( v19 && v4 )
  {
    if ( v20 )
      goto LABEL_20;
    goto LABEL_26;
  }
  return 0;
}
