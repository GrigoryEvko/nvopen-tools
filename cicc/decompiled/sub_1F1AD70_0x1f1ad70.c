// Function: sub_1F1AD70
// Address: 0x1f1ad70
//
__int64 __fastcall sub_1F1AD70(_QWORD *a1, int a2, int *a3, __int64 a4, __int64 a5, unsigned __int64 *a6)
{
  __int64 v9; // r8
  unsigned __int64 v10; // rcx
  int v11; // r10d
  unsigned int v12; // edx
  __int64 v13; // rsi
  __int64 v14; // r15
  __int64 v15; // rbx
  int v16; // edi
  __int64 v17; // r9
  __int64 v18; // r15
  __int64 *v19; // rdx
  __int32 v20; // r15d
  __int64 v21; // rax
  int v22; // ecx
  unsigned __int64 v23; // rcx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rdi
  unsigned int v28; // r15d
  __int64 v29; // rsi
  unsigned int v30; // ebx
  __int64 v31; // rsi
  __int64 v32; // rdx
  _QWORD *v33; // rdi
  _QWORD *v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdx
  _QWORD *v37; // rdi
  _QWORD *v38; // rdx
  __int64 v39; // rcx
  int v40; // [rsp+Ch] [rbp-74h]
  _QWORD *v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  int v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  unsigned __int8 v48; // [rsp+30h] [rbp-50h]
  _QWORD *v49; // [rsp+30h] [rbp-50h]
  int *v51; // [rsp+40h] [rbp-40h] BYREF
  __int64 v52; // [rsp+48h] [rbp-38h]

  v9 = a1[2];
  v10 = *(unsigned int *)(v9 + 408);
  v11 = *(_DWORD *)(**(_QWORD **)(a1[9] + 16LL) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64LL) + a2));
  v12 = v11 & 0x7FFFFFFF;
  v13 = v11 & 0x7FFFFFFF;
  v14 = 8 * v13;
  if ( (v11 & 0x7FFFFFFFu) >= (unsigned int)v10 || (v15 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8LL * v12)) == 0 )
  {
    v30 = v12 + 1;
    if ( (unsigned int)v10 < v12 + 1 )
    {
      v36 = v30;
      if ( v30 < v10 )
      {
        *(_DWORD *)(v9 + 408) = v30;
      }
      else if ( v30 > v10 )
      {
        if ( v30 > (unsigned __int64)*(unsigned int *)(v9 + 412) )
        {
          v43 = v11;
          v45 = a1[2];
          sub_16CD150(v9 + 400, (const void *)(v9 + 416), v30, 8, v9, (int)a6);
          v9 = v45;
          v11 = v43;
          v36 = v30;
          v10 = *(unsigned int *)(v45 + 408);
        }
        v31 = *(_QWORD *)(v9 + 400);
        v37 = (_QWORD *)(v31 + 8 * v36);
        v38 = (_QWORD *)(v31 + 8 * v10);
        v39 = *(_QWORD *)(v9 + 416);
        if ( v37 != v38 )
        {
          do
            *v38++ = v39;
          while ( v37 != v38 );
          v31 = *(_QWORD *)(v9 + 400);
        }
        *(_DWORD *)(v9 + 408) = v30;
        goto LABEL_24;
      }
    }
    v31 = *(_QWORD *)(v9 + 400);
LABEL_24:
    v49 = (_QWORD *)v9;
    *(_QWORD *)(v31 + v14) = sub_1DBA290(v11);
    v15 = *(_QWORD *)(v49[50] + v14);
    sub_1DBB110(v49, v15);
    v9 = a1[2];
    v10 = *(unsigned int *)(v9 + 408);
    v11 = *(_DWORD *)(**(_QWORD **)(a1[9] + 16LL) + 4LL * (unsigned int)(*(_DWORD *)(a1[9] + 64LL) + a2));
    v12 = v11 & 0x7FFFFFFF;
    v13 = v11 & 0x7FFFFFFF;
  }
  v48 = a2 != 0;
  v16 = *(_DWORD *)(*(_QWORD *)(a1[3] + 312LL) + 4 * v13);
  if ( v16 )
  {
    v11 = *(_DWORD *)(*(_QWORD *)(a1[3] + 312LL) + 4 * v13);
    v12 = v16 & 0x7FFFFFFF;
    v13 = v16 & 0x7FFFFFFF;
  }
  v17 = 8 * v13;
  if ( (unsigned int)v10 <= v12 || (v18 = *(_QWORD *)(*(_QWORD *)(v9 + 400) + 8 * v13)) == 0 )
  {
    v28 = v12 + 1;
    if ( v12 + 1 > (unsigned int)v10 )
    {
      v32 = v28;
      if ( v28 < v10 )
      {
        *(_DWORD *)(v9 + 408) = v28;
      }
      else if ( v28 > v10 )
      {
        if ( v28 > (unsigned __int64)*(unsigned int *)(v9 + 412) )
        {
          v40 = v11;
          v42 = v9;
          sub_16CD150(v9 + 400, (const void *)(v9 + 416), v28, 8, v9, v17);
          v9 = v42;
          v17 = 8 * v13;
          v11 = v40;
          v32 = v28;
          v10 = *(unsigned int *)(v42 + 408);
        }
        v29 = *(_QWORD *)(v9 + 400);
        v33 = (_QWORD *)(v29 + 8 * v32);
        v34 = (_QWORD *)(v29 + 8 * v10);
        v35 = *(_QWORD *)(v9 + 416);
        if ( v33 != v34 )
        {
          do
            *v34++ = v35;
          while ( v33 != v34 );
          v29 = *(_QWORD *)(v9 + 400);
        }
        *(_DWORD *)(v9 + 408) = v28;
        goto LABEL_21;
      }
    }
    v29 = *(_QWORD *)(v9 + 400);
LABEL_21:
    v41 = (_QWORD *)v9;
    v44 = v17;
    *(_QWORD *)(v29 + v17) = sub_1DBA290(v11);
    v18 = *(_QWORD *)(v41[50] + v44);
    sub_1DBB110(v41, v18);
  }
  v19 = (__int64 *)sub_1DB3C70((__int64 *)v18, a4);
  if ( v19 == (__int64 *)(*(_QWORD *)v18 + 24LL * *(unsigned int *)(v18 + 8)) )
  {
    v20 = *(_DWORD *)(v15 + 112);
  }
  else
  {
    v20 = *(_DWORD *)(v15 + 112);
    if ( (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) <= (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(a4 >> 1)
                                                                                             & 3) )
    {
      v25 = v19[2];
      if ( v25 )
      {
        v52 = 0;
        v26 = 0;
        v51 = a3;
        if ( (*(_QWORD *)(v25 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          v26 = *(_QWORD *)((*(_QWORD *)(v25 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
        v27 = a1[9];
        v52 = v26;
        if ( (unsigned __int8)sub_2100320(v27, &v51, v25, a4, 1) )
        {
          v23 = sub_21024A0(a1[9], a5, (_DWORD)a6, v20, (unsigned int)&v51, a1[7], v48);
          return sub_1F1A750((__int64)a1, a2, a3, v23, 0);
        }
      }
    }
  }
  v21 = *(_QWORD *)(v15 + 104);
  v22 = -1;
  if ( v21 )
  {
    v22 = 0;
    do
    {
      v22 |= *(_DWORD *)(v21 + 112);
      v21 = *(_QWORD *)(v21 + 104);
    }
    while ( v21 );
  }
  v23 = sub_1F19E60(a1, *(_DWORD *)(*(_QWORD *)(a1[9] + 8LL) + 112LL), v20, v22, a5, a6, v48, a2);
  return sub_1F1A750((__int64)a1, a2, a3, v23, 0);
}
