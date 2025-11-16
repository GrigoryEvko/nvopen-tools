// Function: sub_1F200F0
// Address: 0x1f200f0
//
void __fastcall sub_1F200F0(__int64 a1, unsigned int a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r11
  __int64 v10; // r14
  __int64 *v11; // rax
  __int64 v12; // r15
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rax
  __int64 v19; // r8
  int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r8
  int v26; // r9d
  __int64 v27; // [rsp+0h] [rbp-50h]
  int v28; // [rsp+Ch] [rbp-44h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 256LL) + 96LL) + 8LL * a2);
  if ( (_DWORD)a5 )
  {
    if ( a3 )
    {
      v10 = a1 + 200;
      v11 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL) + 392LL) + 16LL * a2);
      v12 = *v11;
      v30 = v11[1];
      if ( (_DWORD)a5 == a3 && ((a6 | a4) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        *(_DWORD *)(a1 + 80) = a5;
        v23 = v30;
        v22 = (unsigned int)a5;
        goto LABEL_13;
      }
      v13 = *(_QWORD *)a1;
      v14 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)a1 + 96LL) + 8LL * a2);
      v15 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 16LL * *(unsigned int *)(v14 + 48));
      if ( (*v15 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v15[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v27 = a6;
        v28 = a5;
        v29 = v8;
        v16 = sub_1F13A50((_QWORD *)(v13 + 48), *(_QWORD *)(v13 + 40), v14, *v15, a5, a6);
        a6 = v27;
        LODWORD(a5) = v28;
        v8 = v29;
      }
      else
      {
        v16 = *v15;
      }
      if ( (_DWORD)a5 == a3 )
        goto LABEL_12;
      v17 = a4 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (a6 & 0xFFFFFFFFFFFFFFF8LL) != 0
          && *(_DWORD *)(v17 + 24) <= (*(_DWORD *)((a6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | 3u) )
        {
LABEL_12:
          *(_DWORD *)(a1 + 80) = a5;
          v18 = sub_1F1B250(a1, a6);
          sub_1F1FA40(v10, v18, v30, *(unsigned int *)(a1 + 80), v19, v20);
          *(_DWORD *)(a1 + 80) = a3;
          v21 = sub_1F1B330((_QWORD *)a1, a4);
          v22 = *(unsigned int *)(a1 + 80);
          v23 = v21;
LABEL_13:
          sub_1F1FA40(v10, v12, v23, v22, a5, a6);
          return;
        }
        *(_DWORD *)(a1 + 80) = a5;
        if ( (*(_DWORD *)(v17 + 24) | (unsigned int)(a4 >> 1) & 3) < (*(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                    | (unsigned int)(v16 >> 1) & 3) )
        {
          v24 = sub_1F1B1B0(a1, a4);
          sub_1F1FA40(v10, v24, v30, *(unsigned int *)(a1 + 80), v25, v26);
          goto LABEL_17;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 80) = a5;
      }
      v24 = sub_1F1FC90((int *)a1, v8);
LABEL_17:
      *(_DWORD *)(a1 + 80) = a3;
      v22 = a3;
      v23 = v24;
      goto LABEL_13;
    }
    *(_DWORD *)(a1 + 80) = a5;
    sub_1F1FC90((int *)a1, v8);
  }
  else
  {
    *(_DWORD *)(a1 + 80) = a3;
    sub_1F20000(a1, v8);
  }
}
