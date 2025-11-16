// Function: sub_2F73FB0
// Address: 0x2f73fb0
//
__int64 __fastcall sub_2F73FB0(
        __int64 a1,
        __int64 a2,
        char a3,
        int a4,
        __int64 a5,
        unsigned __int8 (__fastcall *a6)(__int64, __int64, __int64),
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rdx
  __int64 v16; // r10
  unsigned int v17; // eax
  __int64 v18; // r9
  unsigned int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // r8
  __int64 *v22; // rax
  __int64 v23; // r8
  _QWORD *v24; // r13
  __int64 v25; // r12
  __int64 v26; // rbx
  unsigned __int64 v27; // r11
  __int64 v28; // rax
  __int64 v29; // r8
  _QWORD *v30; // rcx
  _QWORD *v31; // rdi
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+10h] [rbp-40h]
  __int64 *v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  result = a7;
  if ( a4 < 0 )
  {
    v15 = *(unsigned int *)(a1 + 160);
    v16 = a2;
    v17 = a4 & 0x7FFFFFFF;
    v18 = 8LL * (a4 & 0x7FFFFFFF);
    if ( (a4 & 0x7FFFFFFFu) < (unsigned int)v15 )
    {
      v23 = *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8LL * v17);
      if ( v23 )
      {
LABEL_9:
        if ( a3 )
        {
          v24 = *(_QWORD **)(v23 + 104);
          if ( v24 )
          {
            v25 = 0;
            v26 = 0;
            do
            {
              if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64))a6)(v24, a5) )
              {
                v26 |= v24[14];
                v25 |= v24[15];
              }
              v24 = (_QWORD *)v24[13];
            }
            while ( v24 );
            return v26;
          }
          v37 = v16;
          if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))a6)(v23, a5) )
            return sub_2EBF1E0(v37, a4);
        }
        else if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))a6)(v23, a5) )
        {
          return -1;
        }
        return 0;
      }
    }
    v19 = v17 + 1;
    if ( (unsigned int)v15 < v19 )
    {
      v27 = v19;
      if ( v19 != v15 )
      {
        if ( v19 >= v15 )
        {
          v28 = *(_QWORD *)(a1 + 168);
          v29 = v27 - v15;
          if ( v27 > *(unsigned int *)(a1 + 164) )
          {
            v32 = v27 - v15;
            v34 = *(_QWORD *)(a1 + 168);
            sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v27, 8u, v29, v18);
            v15 = *(unsigned int *)(a1 + 160);
            v16 = a2;
            v29 = v32;
            v28 = v34;
            v18 = 8LL * (a4 & 0x7FFFFFFF);
          }
          v20 = *(_QWORD *)(a1 + 152);
          v30 = (_QWORD *)(v20 + 8 * v15);
          v31 = &v30[v29];
          if ( v30 != v31 )
          {
            do
              *v30++ = v28;
            while ( v31 != v30 );
            LODWORD(v15) = *(_DWORD *)(a1 + 160);
            v20 = *(_QWORD *)(a1 + 152);
          }
          *(_DWORD *)(a1 + 160) = v29 + v15;
          goto LABEL_8;
        }
        *(_DWORD *)(a1 + 160) = v19;
      }
    }
    v20 = *(_QWORD *)(a1 + 152);
LABEL_8:
    v33 = v16;
    v35 = (__int64 *)(v20 + v18);
    v21 = sub_2E10F30(a4);
    v22 = v35;
    v36 = v21;
    *v22 = v21;
    sub_2E11E80((_QWORD *)a1, v21);
    v16 = v33;
    v23 = v36;
    goto LABEL_9;
  }
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 424) + 8LL * (unsigned int)a4);
  if ( v14 )
    return -(__int64)(a6(v14, a5, a8) != 0);
  return result;
}
