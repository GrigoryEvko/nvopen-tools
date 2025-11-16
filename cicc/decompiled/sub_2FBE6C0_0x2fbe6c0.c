// Function: sub_2FBE6C0
// Address: 0x2fbe6c0
//
void __fastcall sub_2FBE6C0(__int64 a1, unsigned int a2, unsigned int a3, __int64 a4, unsigned int a5, __int64 a6)
{
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // r8
  __int64 v11; // r10
  __int64 v12; // r11
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // r14
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // [rsp+0h] [rbp-50h]
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  unsigned __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  unsigned __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  unsigned __int64 v37; // [rsp+18h] [rbp-38h]
  unsigned __int64 v38; // [rsp+18h] [rbp-38h]

  v9 = (unsigned __int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 152LL) + 16LL * a2);
  v10 = *v9;
  v11 = v9[1];
  v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) + 96LL) + 8LL * a2);
  if ( a5 )
  {
    if ( a3 )
    {
      v14 = a1 + 192;
      if ( a5 == a3 && ((a4 | a6) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        *(_DWORD *)(a1 + 80) = a5;
        v24 = a5;
        v25 = v11;
        goto LABEL_13;
      }
      v15 = *(_QWORD *)a1;
      v16 = *(_QWORD *)(*(_QWORD *)(**(_QWORD **)a1 + 96LL) + 8LL * a2);
      v17 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 16LL * *(unsigned int *)(v16 + 24));
      if ( (*v17 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v17[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v30 = a6;
        v31 = v11;
        v33 = v10;
        v36 = v12;
        v18 = sub_2FB0650((_QWORD *)(v15 + 48), *(_QWORD *)(v15 + 40), v16, *v17, v10);
        a6 = v30;
        v11 = v31;
        v10 = v33;
        v12 = v36;
      }
      else
      {
        v18 = *v17;
      }
      if ( a5 == a3 )
        goto LABEL_12;
      v19 = a4 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (a4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        if ( (a6 & 0xFFFFFFFFFFFFFFF8LL) != 0
          && *(_DWORD *)(v19 + 24) <= (*(_DWORD *)((a6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | 3u) )
        {
LABEL_12:
          *(_DWORD *)(a1 + 80) = a5;
          v35 = v10;
          v32 = v11;
          v20 = sub_2FBA660(a1, a6);
          sub_2FBD6E0(v14, v20, v32, *(unsigned int *)(a1 + 80), v21, v22);
          *(_DWORD *)(a1 + 80) = a3;
          v23 = sub_2FBA8B0((__int64 *)a1, a4);
          v24 = *(unsigned int *)(a1 + 80);
          v10 = v35;
          v25 = v23;
LABEL_13:
          sub_2FBD6E0(v14, v10, v25, v24, v10, a6);
          return;
        }
        *(_DWORD *)(a1 + 80) = a5;
        if ( (*(_DWORD *)(v19 + 24) | (unsigned int)(a4 >> 1) & 3) < (*(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                    | (unsigned int)(v18 >> 1) & 3) )
        {
          v38 = v10;
          v34 = v11;
          v27 = sub_2FBA5C0(a1, a4);
          sub_2FBD6E0(v14, v27, v34, *(unsigned int *)(a1 + 80), v28, v29);
          v10 = v38;
          goto LABEL_17;
        }
      }
      else
      {
        *(_DWORD *)(a1 + 80) = a5;
      }
      v37 = v10;
      v26 = sub_2FBDDD0((_QWORD *)a1, v12);
      v10 = v37;
      v27 = v26;
LABEL_17:
      *(_DWORD *)(a1 + 80) = a3;
      v24 = a3;
      v25 = v27;
      goto LABEL_13;
    }
    *(_DWORD *)(a1 + 80) = a5;
    sub_2FBDDD0((_QWORD *)a1, v12);
  }
  else
  {
    *(_DWORD *)(a1 + 80) = a3;
    sub_2FBE460(a1, v12);
  }
}
