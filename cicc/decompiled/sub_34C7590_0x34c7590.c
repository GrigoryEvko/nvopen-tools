// Function: sub_34C7590
// Address: 0x34c7590
//
__int64 __fastcall sub_34C7590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // r13d
  __int64 *v8; // rsi
  int v9; // eax
  __int64 v10; // rax
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // r15
  __int16 v15; // ax
  __int64 (__fastcall *v16)(__int64); // rcx
  _DWORD *v17; // rax
  _DWORD *v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // rcx
  int v22; // edx
  unsigned __int64 v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r12
  unsigned __int64 v26; // rbx
  __int64 *v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned int v30; // eax
  __int64 v31; // rcx
  __int64 *v32; // rbx
  __int64 v33; // rax
  __int64 v34; // r12
  unsigned __int64 v35; // r10
  _QWORD *v36; // rax
  _QWORD *v37; // rsi
  unsigned __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 *v39; // [rsp+10h] [rbp-80h]
  __int64 *v40; // [rsp+20h] [rbp-70h]
  int v42; // [rsp+34h] [rbp-5Ch]
  _QWORD v44[2]; // [rsp+40h] [rbp-50h] BYREF
  char v45; // [rsp+50h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 112);
  v8 = *(__int64 **)(a1 + 64);
  v40 = v8;
  v9 = *(_DWORD *)(*(_QWORD *)(a3 + 80) + 4LL * (v7 & 0x7FFFFFFF));
  if ( !v9 )
    v9 = *(_DWORD *)(a1 + 112);
  v42 = v9;
  v39 = &v8[*(unsigned int *)(a1 + 72)];
  if ( v8 == v39 )
    return 1;
LABEL_4:
  v10 = *(_QWORD *)(*v40 + 8);
  if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_5;
  if ( (v10 & 6) != 0 )
  {
    v12 = *(_QWORD *)((*(_QWORD *)(*v40 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v13 = *v40;
    v14 = a2;
    while ( 1 )
    {
      v15 = *(_WORD *)(v12 + 68);
      if ( v15 == 20 )
      {
        v17 = *(_DWORD **)(v12 + 32);
        v18 = v17 + 10;
      }
      else
      {
        v16 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a4 + 520LL);
        if ( v16 == sub_2DCA430 )
        {
          a2 = v14;
          goto LABEL_12;
        }
        ((void (__fastcall *)(_QWORD *, __int64, __int64))v16)(v44, a4, v12);
        v17 = (_DWORD *)v44[0];
        v18 = (_DWORD *)v44[1];
        if ( !v45 )
        {
LABEL_18:
          v15 = *(_WORD *)(v12 + 68);
          a2 = v14;
LABEL_12:
          if ( (v15 != 10 || (*(_DWORD *)(v12 + 40) & 0xFFFFFF) != 1)
            && ((*(_BYTE *)(*(_QWORD *)(v12 + 16) + 27LL) & 0x20) == 0
             || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)a4 + 56LL))(a4, v12)) )
          {
            return 0;
          }
LABEL_5:
          if ( v39 == ++v40 )
            return 1;
          goto LABEL_4;
        }
      }
      if ( (*v17 & 0xFFF00) != 0 || (*v18 & 0xFFF00) != 0 )
        goto LABEL_18;
      v19 = *(_QWORD *)(v12 + 32);
      if ( *(_DWORD *)(v19 + 8) != v7 )
        return 0;
      v7 = *(_DWORD *)(v19 + 48);
      if ( v7 >= 0 )
        return 0;
      v20 = v7 & 0x7FFFFFFF;
      v21 = v7 & 0x7FFFFFFF;
      v22 = *(_DWORD *)(*(_QWORD *)(a3 + 80) + 4 * v21);
      if ( !v22 )
        v22 = v7;
      if ( v22 != v42 )
        return 0;
      v23 = *(unsigned int *)(v14 + 160);
      v24 = 8 * v21;
      if ( v20 >= (unsigned int)v23 )
        break;
      v25 = *(_QWORD *)(*(_QWORD *)(v14 + 152) + 8 * v21);
      if ( !v25 )
        break;
LABEL_26:
      v26 = *(_QWORD *)(v13 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      v27 = (__int64 *)sub_2E09D00((__int64 *)v25, v26);
      v28 = *(_QWORD *)v25 + 24LL * *(unsigned int *)(v25 + 8);
      if ( v27 == (__int64 *)v28 )
        BUG();
      if ( (*(_DWORD *)((*v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v27 >> 1) & 3) > *(_DWORD *)(v26 + 24)
        || ((v13 = v27[2], v29 = *(_QWORD *)(v13 + 8), v26 != (v27[1] & 0xFFFFFFFFFFFFFFF8LL))
         || (__int64 *)v28 != v27 + 3)
        && v26 == v29 )
      {
        v29 = MEMORY[8];
        v13 = 0;
      }
      if ( (v29 & 6) == 0 )
        return 0;
      v12 = *(_QWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    }
    v30 = v20 + 1;
    if ( (unsigned int)v23 < v30 && v30 != v23 )
    {
      if ( v30 >= v23 )
      {
        v34 = *(_QWORD *)(v14 + 168);
        v35 = v30 - v23;
        if ( v30 > (unsigned __int64)*(unsigned int *)(v14 + 164) )
        {
          v38 = v30 - v23;
          sub_C8D5F0(v14 + 152, (const void *)(v14 + 168), v30, 8u, a5, a6);
          v23 = *(unsigned int *)(v14 + 160);
          v35 = v38;
        }
        v31 = *(_QWORD *)(v14 + 152);
        v36 = (_QWORD *)(v31 + 8 * v23);
        v37 = &v36[v35];
        if ( v36 != v37 )
        {
          do
            *v36++ = v34;
          while ( v37 != v36 );
          LODWORD(v23) = *(_DWORD *)(v14 + 160);
          v31 = *(_QWORD *)(v14 + 152);
        }
        *(_DWORD *)(v14 + 160) = v35 + v23;
        goto LABEL_34;
      }
      *(_DWORD *)(v14 + 160) = v30;
    }
    v31 = *(_QWORD *)(v14 + 152);
LABEL_34:
    v32 = (__int64 *)(v31 + v24);
    v33 = sub_2E10F30(v7);
    *v32 = v33;
    v25 = v33;
    sub_2E11E80((_QWORD *)v14, v33);
    goto LABEL_26;
  }
  return 0;
}
