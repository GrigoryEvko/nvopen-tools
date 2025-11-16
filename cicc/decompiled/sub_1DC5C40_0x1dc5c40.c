// Function: sub_1DC5C40
// Address: 0x1dc5c40
//
void __fastcall sub_1DC5C40(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 *a5, __int64 a6)
{
  __int64 v11; // r11
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r9
  char v17; // dl
  __int64 v18; // rsi
  _QWORD *v19; // rdi
  __int64 *v20; // rax
  int v21; // r10d
  __int64 *v22; // rdx
  unsigned __int64 v23; // [rsp+0h] [rbp-60h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+10h] [rbp-50h]
  __int64 v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v11 = a1[2];
  v12 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( ((a3 >> 1) & 3) != 0 )
  {
    v14 = (2LL * (int)(((a3 >> 1) & 3) - 1)) | v12;
    v13 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    v13 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
    v14 = v13 | 6;
  }
  if ( !v13 || (v15 = *(_QWORD *)(v13 + 16)) == 0 )
  {
    v18 = *(unsigned int *)(v11 + 544);
    v19 = *(_QWORD **)(v11 + 536);
    v23 = v13;
    v28[0] = v14;
    v24 = v14;
    v26 = (__int64)&v19[2 * v18];
    v20 = sub_1DC32D0(v19, v26, v28);
    v22 = v20;
    if ( (__int64 *)v26 == v20 )
    {
      if ( !v21 )
        goto LABEL_11;
    }
    else if ( (*(_DWORD *)((*v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v20 >> 1) & 3)) <= (*(_DWORD *)(v23 + 24) | (unsigned int)(v24 >> 1) & 3) )
    {
LABEL_11:
      v16 = v22[1];
      goto LABEL_6;
    }
    v22 = v20 - 2;
    goto LABEL_11;
  }
  v16 = *(_QWORD *)(v15 + 24);
LABEL_6:
  v25 = v16;
  if ( !sub_1DB7C30(a2, a5, a6, *(_QWORD *)(*(_QWORD *)(v11 + 392) + 16LL * *(unsigned int *)(v16 + 48)), a3)
    && !v17
    && !(unsigned __int8)sub_1DC5250(a1, a2, v25, a3, a4, v25, a5, a6) )
  {
    sub_1DC4840(a1, a2);
  }
}
