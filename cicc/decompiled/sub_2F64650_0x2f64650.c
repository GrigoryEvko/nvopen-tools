// Function: sub_2F64650
// Address: 0x2f64650
//
__int64 __fastcall sub_2F64650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  _DWORD *v12; // rax
  __int64 v13; // r8
  unsigned __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 *v16; // rdx
  __int64 v17; // r14
  unsigned __int64 v18; // rbx
  __int64 *v19; // rcx
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rsi
  __int64 v23; // rax
  unsigned __int64 v24; // rbx
  __int64 v25; // rsi
  __int128 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rsi
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // r11
  _QWORD *v33; // rdx
  _QWORD *v34; // rdi
  unsigned __int64 v35; // [rsp+0h] [rbp-50h]
  _QWORD *v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+8h] [rbp-48h]
  __int64 *v38; // [rsp+10h] [rbp-40h]
  __int64 *v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  int v42; // [rsp+1Ch] [rbp-34h]

  v6 = a2;
  v7 = *(_QWORD *)(a2 + 8);
  if ( (v7 & 6) == 0 )
    return v6;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v6 + 8);
    v10 = *(_QWORD *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( *(_WORD *)(v10 + 68) != 20 )
      return v6;
    v12 = *(_DWORD **)(v10 + 32);
    if ( (*v12 & 0xFFF00) != 0 )
      return v6;
    if ( (v12[10] & 0xFFF00) != 0 )
      return v6;
    v42 = v12[12];
    if ( v42 >= 0 )
      return v6;
    v13 = *(_QWORD *)(a1 + 56);
    v14 = *(unsigned int *)(v13 + 160);
    v15 = v42 & 0x7FFFFFFF;
    if ( (v42 & 0x7FFFFFFFu) >= (unsigned int)v14 || (v16 = *(__int64 **)(*(_QWORD *)(v13 + 152) + 8LL * v15)) == 0 )
    {
      v21 = v15 + 1;
      if ( (unsigned int)v14 < v21 )
      {
        v30 = v21;
        if ( v21 != v14 )
        {
          if ( v21 >= v14 )
          {
            v31 = *(_QWORD *)(v13 + 168);
            v32 = v30 - v14;
            if ( v30 > *(unsigned int *)(v13 + 164) )
            {
              v35 = v30 - v14;
              v37 = *(_QWORD *)(v13 + 168);
              v41 = *(_QWORD *)(a1 + 56);
              sub_C8D5F0(v13 + 152, (const void *)(v13 + 168), v30, 8u, v13, a6);
              v13 = v41;
              v32 = v35;
              v31 = v37;
              v14 = *(unsigned int *)(v41 + 160);
            }
            v22 = *(_QWORD *)(v13 + 152);
            v33 = (_QWORD *)(v22 + 8 * v14);
            v34 = &v33[v32];
            if ( v33 != v34 )
            {
              do
                *v33++ = v31;
              while ( v34 != v33 );
              LODWORD(v14) = *(_DWORD *)(v13 + 160);
              v22 = *(_QWORD *)(v13 + 152);
            }
            *(_DWORD *)(v13 + 160) = v32 + v14;
            goto LABEL_20;
          }
          *(_DWORD *)(v13 + 160) = v21;
        }
      }
      v22 = *(_QWORD *)(v13 + 152);
LABEL_20:
      v36 = (_QWORD *)v13;
      v23 = sub_2E10F30(v42);
      *(_QWORD *)(v22 + 8LL * (v42 & 0x7FFFFFFF)) = v23;
      v39 = (__int64 *)v23;
      sub_2E11E80(v36, v23);
      v16 = v39;
    }
    if ( *(_BYTE *)(a1 + 32) )
    {
      v17 = v16[13];
      if ( v17 )
      {
        v40 = 0;
        v24 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        while ( 1 )
        {
          v25 = *(unsigned int *)(a1 + 12);
          a6 = *(_QWORD *)(v17 + 120);
          if ( (_DWORD)v25 )
          {
            *(_QWORD *)&v26 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 72) + 312LL))(
                                *(_QWORD *)(a1 + 72),
                                v25,
                                *(_QWORD *)(v17 + 112),
                                *(_QWORD *)(v17 + 120));
            if ( (*(_OWORD *)(a1 + 16) & v26) == 0 )
              goto LABEL_23;
          }
          else if ( (*(_OWORD *)(a1 + 16) & *(_OWORD *)(v17 + 112)) == 0 )
          {
            goto LABEL_23;
          }
          v27 = (__int64 *)sub_2E09D00((__int64 *)v17, v24);
          v28 = *(_QWORD *)v17 + 24LL * *(unsigned int *)(v17 + 8);
          if ( v27 == (__int64 *)v28 )
            goto LABEL_23;
          v29 = 0;
          if ( (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) <= *(_DWORD *)(v24 + 24) )
          {
            v29 = v27[2];
            if ( (v24 != (v27[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v28 != v27 + 3) && v24 == *(_QWORD *)(v29 + 8) )
              v29 = 0;
          }
          if ( !v40 )
          {
            v40 = v29;
            goto LABEL_23;
          }
          if ( !v29 )
          {
LABEL_23:
            v17 = *(_QWORD *)(v17 + 104);
            if ( !v17 )
              goto LABEL_35;
          }
          else
          {
            if ( v29 != v40 )
              return v6;
            v17 = *(_QWORD *)(v17 + 104);
            if ( !v17 )
            {
LABEL_35:
              v6 = v40;
              goto LABEL_15;
            }
          }
        }
      }
    }
    v18 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    v38 = v16;
    v19 = (__int64 *)sub_2E09D00(v16, v18);
    v20 = *v38 + 24LL * *((unsigned int *)v38 + 2);
    if ( v19 == (__int64 *)v20
      || (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) > *(_DWORD *)(v18 + 24) )
    {
      return 0;
    }
    v6 = v19[2];
    if ( (v18 != (v19[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v20 != v19 + 3) && v18 == *(_QWORD *)(v6 + 8) )
      return 0;
LABEL_15:
    if ( !v6 )
      return 0;
    v7 = *(_QWORD *)(v6 + 8);
    if ( (v7 & 6) == 0 )
      return v6;
  }
}
