// Function: sub_145D330
// Address: 0x145d330
//
__int64 __fastcall sub_145D330(__int64 a1, _QWORD *a2, __int64 a3, __int64 *a4, __int64 a5, unsigned int a6)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // rax
  char v16; // r10
  unsigned int v17; // r9d
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r15
  __int64 v23; // rdx
  char v24; // di
  unsigned int v25; // esi
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // rbx
  int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned int v33; // ebx
  __int64 v34; // rax
  __int64 v35; // r14
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // [rsp+4h] [rbp-8Ch]
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+20h] [rbp-70h]
  __int64 v41; // [rsp+20h] [rbp-70h]
  __int64 v42; // [rsp+20h] [rbp-70h]
  char v43; // [rsp+2Bh] [rbp-65h]
  unsigned int v45; // [rsp+3Ch] [rbp-54h] BYREF
  __int64 v46; // [rsp+40h] [rbp-50h] BYREF
  int v47; // [rsp+48h] [rbp-48h]
  __int64 v48; // [rsp+50h] [rbp-40h] BYREF
  int v49; // [rsp+58h] [rbp-38h]

  if ( *((_BYTE *)a4 + 16) != 13 )
    goto LABEL_7;
  v11 = sub_13FCB50(a5);
  if ( !v11 )
    goto LABEL_7;
  v40 = v11;
  v12 = sub_13FC470(a5);
  if ( !v12 )
    goto LABEL_7;
  v39 = v12;
  v43 = 0;
  if ( sub_1454930(a3, &v46, &v45) )
  {
    a3 = v46;
    v43 = 1;
    v38 = v45;
  }
  if ( *(_BYTE *)(a3 + 16) != 77 || *(_QWORD *)(a3 + 40) != **(_QWORD **)(a5 + 32) )
    goto LABEL_7;
  v15 = 0x17FFFFFFE8LL;
  v16 = *(_BYTE *)(a3 + 23) & 0x40;
  v17 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( v17 )
  {
    v18 = 24LL * *(unsigned int *)(a3 + 56) + 8;
    v19 = 0;
    do
    {
      v20 = a3 - 24LL * v17;
      if ( v16 )
        v20 = *(_QWORD *)(a3 - 8);
      if ( v40 == *(_QWORD *)(v20 + v18) )
      {
        v15 = 24 * v19;
        goto LABEL_17;
      }
      ++v19;
      v18 += 8;
    }
    while ( v17 != (_DWORD)v19 );
    v15 = 0x17FFFFFFE8LL;
  }
LABEL_17:
  v21 = v16 ? *(_QWORD *)(a3 - 8) : a3 - 24LL * v17;
  if ( !sub_1454930(*(_QWORD *)(v21 + v15), &v46, &v45) || v46 != a3 || v43 && v45 != v38 )
    goto LABEL_7;
  v22 = sub_1632FA0(*(_QWORD *)(a2[3] + 40LL));
  if ( v45 <= 0x18 )
  {
    v31 = sub_159C470(*a4, 0, 0);
  }
  else
  {
    v23 = 0x17FFFFFFE8LL;
    v24 = *(_BYTE *)(a3 + 23) & 0x40;
    v25 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
    if ( v25 )
    {
      v26 = 24LL * *(unsigned int *)(a3 + 56) + 8;
      v27 = 0;
      do
      {
        v28 = a3 - 24LL * v25;
        if ( v24 )
          v28 = *(_QWORD *)(a3 - 8);
        if ( v39 == *(_QWORD *)(v28 + v26) )
        {
          v23 = 24 * v27;
          goto LABEL_31;
        }
        ++v27;
        v26 += 8;
      }
      while ( v25 != (_DWORD)v27 );
      v23 = 0x17FFFFFFE8LL;
    }
LABEL_31:
    v29 = v24 ? *(_QWORD *)(a3 - 8) : a3 - 24LL * v25;
    v41 = v23;
    v30 = sub_157EBA0(v39);
    sub_14C2530((unsigned int)&v46, *(_QWORD *)(v29 + v41), v22, 0, 0, v30, a2[7], 0);
    v42 = *a4;
    if ( sub_13D0200(&v46, v47 - 1) )
    {
      v31 = sub_159C470(v42, 0, 0);
    }
    else
    {
      if ( !sub_13D0200(&v48, v49 - 1) )
      {
        v37 = sub_1456E90((__int64)a2);
        sub_14573F0(a1, v37);
        sub_135E100(&v48);
        sub_135E100(&v46);
        return a1;
      }
      v31 = sub_159C470(v42, -1, 1);
    }
    sub_135E100(&v48);
    sub_135E100(&v46);
  }
  v32 = sub_14D7760(a6, v31, a4, v22, a2[5]);
  if ( (unsigned __int8)sub_1595F50(v32) )
  {
    v33 = sub_1456C90((__int64)a2, *a4);
    v34 = sub_1456E10((__int64)a2, *a4);
    v35 = sub_145CF80((__int64)a2, v34, v33, 0);
    v36 = sub_1456E90((__int64)a2);
    sub_14575D0(a1, v36, v35, 0);
  }
  else
  {
LABEL_7:
    v13 = sub_1456E90((__int64)a2);
    sub_14573F0(a1, v13);
  }
  return a1;
}
